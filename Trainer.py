import torch
import torch.nn as nn
import time
from CustomDataClass import TextData
import optuna
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score

class Trainer: 
    """The base class for training models with data."""
    def __init__(self, max_epochs=1, batch_size=36, early_stopping_patience=6, 
                 min_delta=0.09, num_gpus=0, max_length=16):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_length = max_length

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    def prepare_training_data(self, text_data):
        self.train_dataloader = DataLoader(text_data, batch_size=self.batch_size, shuffle=True)
    
    def prepare_test_data(self, test_dataset):
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
            
    def prepare_model(self, model):
        model.trainer = self
        self.model = model.to(self.device)  # Move model to the device
    
    def fit(self, model, dataset):
        self.train_loss_values = []
        self.val_loss_values = []
        self.prepare_training_data(dataset)
        self.prepare_model(model)
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = self.fit_epoch()

    def fit_epoch(self):
        train_loss = 0.0
        total_batches = len(self.train_dataloader)

        for idx, (text, attention_mask, labels) in enumerate(self.train_dataloader):
            # Move inputs to the device
            text = text.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            output, _ = self.model(text, attention_mask)
            loss = self.model.loss(output, labels)
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            train_loss += loss.item() * text.size(0)

            # Print progress
            progress = (idx + 1) / total_batches * 100
            print(f"\rBatch {idx + 1}/{total_batches} completed. Progress: {progress:.2f}%", end='', flush=True)

        train_loss /= len(self.train_dataloader.dataset)
        return train_loss

    def test(self, model, data):
        model.eval()
        self.prepare_test_data(data)
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for text, attention_mask, labels in self.test_dataloader:
                # Move inputs to the device
                text = text.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                y_hat, _ = model(text, attention_mask)
                probabilities = torch.sigmoid(y_hat)
                all_targets.append(labels)
                all_predictions.append(probabilities)
                print('batch_done') 
        
        all_targets = torch.cat(all_targets).cpu()  # Move to CPU for metrics calculation
        all_predictions = torch.cat(all_predictions).cpu()

        y_true = all_targets.numpy()
        y_pred_prob = all_predictions.numpy() 

        # Metrics calculation
        subset_acc = accuracy_score(y_true, (y_pred_prob > 0.5).astype(int))
        hamming = hamming_loss(y_true, (y_pred_prob > 0.5).astype(int))
        roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')

        return subset_acc, hamming, roc_auc

