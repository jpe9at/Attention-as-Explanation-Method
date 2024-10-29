import pandas as pd
import numpy as np
import argparse
from CustomDataClass import clean_text, TextData
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer
from Module import BERTClassifier
from torch.utils.data import DataLoader
from Trainer import Trainer
import io
import shap
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score
import time
from datasets import load_dataset

##########################################################################
# Path filepath as argument.
parser = argparse.ArgumentParser()

# Step 2: Add the file path argument
#parser.add_argument('file_path', type=str, help='The path to the input file')
parser.add_argument('path_to_save', type=str, help='The path to save the trained model to')
parser.add_argument('--cuda_device', type=int, default=0, help='Specify the CUDA device number (default: 0)')

args = parser.parse_args()

print('Selected device')
print(args.cuda_device)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

save_path = args.path_to_save +'_db_pedia.pth'
#save_path_parameters = args.path_to_save + '_db_pedia_parameters.pth'

###########################################################################

print('Load the dataframe')
dataset = load_dataset("dbpedia_14")



train_dataset = pd.DataFrame(dataset["train"])
test_dataset = pd.DataFrame(dataset["test"])
# Split data into train and test sets
X_train, X_val, y_train, y_val = train_test_split(train_dataset['content'], train_dataset['label'], test_size=0.2, random_state=42)




tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
# Create the dataset
train_dataset = TextData(X_train[0:6000], y_train[0:6000], tokenizer, max_length=25, train=True, labels_datatype = 'long')
val_dataset = TextData(X_val[0:750], y_val[0:750], tokenizer, max_length=25, train=True, labels_datatype = 'long')

test_dataset = TextData(test_dataset['content'][0:1000], test_dataset['label'][0:1000], tokenizer, max_length=25, train=True, labels_datatype = 'long')

###############################################################################
#Train the model
###############################################################################

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = BERTClassifier('prajjwal1/bert-tiny', 14, loss_function = 'CEL').to(device)

# Train the model
trainer = Trainer(max_epochs=25, batch_size = 36)
trainer.fit(model, train_dataset, val_dataset)
accuracy = trainer.test_multiclass(model, test_dataset)
print(f"Subset Accuracy is: {accuracy}")
#print(f"Hamming is: {hamming}")

n_epochs = range(trainer.max_epochs)
train_loss = trainer.train_loss_values
nan_values = np.full(trainer.max_epochs - len(train_loss), np.nan)
train_loss = np.concatenate([train_loss,nan_values])

val_loss = trainer.val_loss_values
nan_values = np.full(trainer.max_epochs - len(val_loss), np.nan)
val_loss = np.concatenate([val_loss,nan_values])

plt.figure(figsize=(10,6))
plt.plot(n_epochs, train_loss, color='blue', label='train_loss' , linestyle='-')
plt.plot(n_epochs, val_loss, color='orange', label='val_loss' , linestyle='-')
plt.title("Train Loss/Val Loss")
plt.legend()

#torch.save(model.state_dict(), save_path_parameters)
torch.save(model, save_path)



# Test explanations on random item.
# Choose instance and label
item = train_dataset.__getitem__(21)
input_ids, attention_mask, labels = item[0].to(device), item[1].to(device), item[2].to(device)

Manual = clean_text("Was this the best I have seen? I think not!")
encoding_manual = tokenizer([Manual], padding=True, truncation=True, return_tensors='pt', max_length=16)

###########################################################################
# Make prediction with explanations
prediction, explanation_scores = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), True)
label_idx = 0

manual_prediction, explanation_manual = model(encoding_manual['input_ids'], encoding_manual['attention_mask'], True)


# Plot Attention-based explanation scores
explanation_scores_label = explanation_scores[0, label_idx, :].detach().cpu().numpy()  # Shape: (seq_len,)
tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu())

manual_explanation_scores_label = explanation_manual[0, label_idx, :].detach().cpu().numpy()  # Shape: (seq_len,)
manual_tokens = tokenizer.convert_ids_to_tokens(encoding_manual['input_ids'].squeeze(0).cpu())

# Plot the Heatmap for Explanation Scores
plt.figure(figsize=(10, 2))
sns.heatmap([explanation_scores_label], annot=[tokens], fmt="", cmap='coolwarm', cbar=True, xticklabels=tokens, yticklabels=["Explanation Scores"])
plt.xlabel("Tokens")
plt.title(f"Explanation Heatmap for Label {label_idx}")
plt.figure(figsize=(10, 2))
sns.heatmap([manual_explanation_scores_label], annot=[manual_tokens], fmt="", cmap='coolwarm', cbar=True, xticklabels=manual_tokens, yticklabels=["Explanation Scores"])
plt.xlabel("Tokens")
plt.title(f"Explanation Heatmap for Label {label_idx}")
###########################################################################

###########################################################################
# Plot SHAP values

# Define a model wrapper function for SHAP
def model_wrapper(sample_inputs):
    sample_encodings = tokenizer(sample_inputs.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=16)
    sample_encodings = {key: tensor.to(device) for key, tensor in sample_encodings.items()}
    output, _ = model(sample_encodings['input_ids'], sample_encodings['attention_mask'])
    return output

# Initialize explainer object
explainer = shap.Explainer(model_wrapper, tokenizer)

# Get SHAP values
decoded_sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
shap_values = explainer([decoded_sentence])  # Shape (batch, length_sequence, label)

shap_values_manual = explainer([Manual])


# Plot the Heatmap for SHAP Values
plt.figure(figsize=(10, 2))
sns.heatmap([shap_values[0, :, 0].values], annot=[tokens], fmt="", cmap='coolwarm', cbar=True, xticklabels=tokens, yticklabels=["Explanation Scores"])
plt.xlabel("Tokens")
plt.title(f"SHAP Heatmap for Label {label_idx}")
plt.figure(figsize=(10, 2))
sns.heatmap([shap_values_manual[0, :, 0].values], annot=[manual_tokens], fmt="", cmap='coolwarm', cbar=True, xticklabels=tokens, yticklabels=["Explanation Scores"])
plt.xlabel("Tokens")
plt.title(f"SHAP Heatmap for Label {label_idx}")
plt.show()


###########################################################################
