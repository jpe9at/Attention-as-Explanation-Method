from transformers import BertModel
from torch import nn, optim
import torch
import torch.nn.init as init
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, device):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.weight_vector = nn.Parameter(torch.randn(hidden_size).to(device))  # Weight vector w_a

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention scores for each word
        scores = torch.matmul(hidden_states, self.weight_vector)  # Shape: (batch_size, seq_len)
        attention_weights = torch.softmax(scores, dim=1)  # Normalize scores into probabilities

        # Expand dimensions for multiplication
        attention_weights = attention_weights.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

        # Compute the context vector as the weighted sum of hidden states
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)  # Shape: (batch_size, hidden_size)

        return context_vector, attention_weights

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, optimizer='SGD', learning_rate=0.001, loss_function='BCE', l1=0.0, l2=0.0, clip_val=0, scheduler=None):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        device = next(self.parameters()).device  # Get current device
        self.attention_layer = AttentionLayer(hidden_size=self.bert.config.hidden_size, device=device)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        self.learning_rate = learning_rate
        self.l1_rate = l1
        self.l2_rate = l2
        if clip_val != 0: 
            self.clip_gradients(clip_val)

        self.optimizer = self.get_optimizer(optimizer, self.learning_rate, self.l2_rate)
        self.loss = self.get_loss(loss_function)

    def forward(self, input_ids, attention_mask, exp_scores=False):
        # Ensure the input tensors are on the correct device
        input_ids = input_ids.to(next(self.parameters()).device)
        attention_mask = attention_mask.to(next(self.parameters()).device)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        hidden_states = outputs.last_hidden_state

        context_vector, attention_weights = self.attention_layer(hidden_states)  # context_vector shape: (batch_size, hidden_size)

        # Pass the context vector through the classification layer
        logits = self.fc(context_vector) 

        explanation_scores = None
        if exp_scores:
            explanation_scores = self.compute_explanation_scores(hidden_states, attention_weights, self.fc.weight.data)

        return logits, explanation_scores

    def compute_explanation_scores(self, hidden_states, attention_weights, classifier_weights):
        # Ensure the weights are on the correct device
        classifier_weights = classifier_weights.to(next(self.parameters()).device)

        # hidden_states shape: (batch_size, seq_len, hidden_size)
        # attention_weights shape: (batch_size, seq_len, 1)
        # classifier_weights shape: (num_labels, hidden_size)

        # Squeeze the singleton dimension from attention_weights
        attention_weights = attention_weights.squeeze(-1)  # Shape: (batch_size, seq_len)

        # Compute the weighted hidden states
        weighted_hidden_states = attention_weights.unsqueeze(-1) * hidden_states  # Shape: (batch_size, seq_len, hidden_size)

        # Compute the influence of classifier weights on each weighted_hidden_state
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_labels = classifier_weights.shape[0]
        explanation_scores = torch.zeros((batch_size, num_labels, seq_len), device=hidden_states.device)  # Shape: (batch_size, num_labels, seq_len)

        for label_idx in range(num_labels):
            # Compute W_classifier * (a_t * h_t) for each label
            classifier_weight = classifier_weights[label_idx]  # Shape: (hidden_size,)
            influence = torch.einsum('bth,h->bt', weighted_hidden_states, classifier_weight)  # Shape: (batch_size, seq_len)

            # Take the sign of the influence and multiply it with attention weights
            explanation_scores[:, label_idx, :] = attention_weights * torch.sign(influence)  # Shape: (batch_size, seq_len)

        return explanation_scores  # Shape: (batch_size, num_labels, seq_len)

    def get_optimizer(self, optimizer, learning_rate, l2):
        Optimizers = {
            'Adam': optim.Adam(self.parameters(), lr=learning_rate, weight_decay=l2),
            'SGD': optim.SGD(self.parameters(), lr=learning_rate, momentum=0.09, weight_decay=l2)
        }
        return Optimizers[optimizer]

    def get_loss(self, loss_function):
        Loss_Functions = {
            'CEL': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'Huber': nn.HuberLoss(),
            'BCE': nn.BCEWithLogitsLoss()
        }
        return Loss_Functions[loss_function]

    def get_scheduler(self, scheduler, optimizer):
        if scheduler is None:
            return None
        schedulers = {
            'OnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.15, patience=4, threshold=0.01)
        }
        return schedulers[scheduler]
