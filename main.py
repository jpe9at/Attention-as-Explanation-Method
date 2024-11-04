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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score
import time
from datasets import load_dataset
import os
import sys
##########################################################################
# Path filepath as argument.
parser = argparse.ArgumentParser()

# Step 2: Add the file path argument
parser.add_argument('path_to_save', type=str, help='The path to save the trained model to')
parser.add_argument('--cuda_device', type=int, default=0, help='Specify the CUDA device number (default: 0)')
parser.add_argument('--dataset', type=str, help='The path to the dataset')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

if not os.path.exists(args.path_to_save):
    print(f"{args.path_to_save} is not a valid path.")
    sys.exit(1)

save_path = args.path_to_save + 'model.pth'

#will be changed to True if using dbpedia dataset
multiclass = False

'''
###########################################################################
#if training on jigsaw
###########################################################################
if args.dataset is None:
    print("Error: --dataset is required")
    sys.exit(1)  # Exit with a non-zero status to indicate an error
else:
    path = args.dataset


print('Load the dataframe')
tweet_dataframe = pd.read_csv(path)

tweet_dataframe['comment_text'] = tweet_dataframe['comment_text'].apply(clean_text)

# Split data into features (X) and labels (y)
X = tweet_dataframe['comment_text']
y = tweet_dataframe.iloc[:, 2:]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
###########################################################################

print('Load the dataframe')

#select dataset, uncomment the desired dataset
#dataset = load_dataset("imdb")

dataset = load_dataset("yelp_polarity")

#dataset = load_dataset("dbpedia_14")
#multiclass = True

train_dataset = pd.DataFrame(dataset["train"])
test_dataset = pd.DataFrame(dataset["test"])

# Split data into train and test sets
X_train, X_val, y_train, y_val = train_test_split(train_dataset['text'], train_dataset['label'], test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

# Create the dataset
train_dataset = TextData(X_train, y_train.to_frame(), tokenizer, max_length=25, train=True)
val_dataset = TextData(X_val, y_val.to_frame(), tokenizer, max_length=25, train=True)
test_dataset = TextData(test_dataset['text'], test_dataset['label'].to_frame(), tokenizer, max_length=25, train=True)

#if using db_pedia
#train_dataset = TextData(X_train, y_train, tokenizer, max_length=25, train=True, labels_datatype = 'long')
#val_dataset = TextData(X_val, y_val, tokenizer, max_length=25, train=True, labels_datatype = 'long')
#test_dataset = TextData(test_dataset['content'], test_dataset['label'], tokenizer, max_length=25, train=True, labels_datatype = 'long')




###############################################################################
#Train the model
###############################################################################
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training on device: {device}')


if multiclass == False: 
    num_outputs = len(test_dataset.labels[0])
    
    #Need to select if attention = 'sparse' or 'dense'
    model = BERTClassifier('prajjwal1/bert-tiny', num_outputs, learning_rate=0.0001, optimizer = 'Adam', loss_function = 'BCE', attention = 'sparse').to(device)

else:
    #hardcoded number of classes for the dbpedia dataset
    #Need to select if attention = 'sparse' or 'dense'
    model = BERTClassifier('prajjwal1/bert-tiny', 14, learning_rate=0.0001, optimizer = 'Adam', loss_function = 'CEL', attention = 'sparse').to(device)


# Train the model
trainer = Trainer(max_epochs=1, batch_size = 64)
trainer.fit(model, train_dataset, val_dataset)

acc, _ = trainer.test(model, test_dataset)    #need to use .test_multiclass for dbpedia
print(f"Accuracy is: {acc}")

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

torch.save(model, save_path)


###########################################################################
# Test explanations on random item.
###########################################################################

###########################################################################
# Choose instance and label from datset 
item = train_dataset.__getitem__(54)
input_ids, attention_mask, labels = item[0].to(device), item[1].to(device), item[2].to(device)


# Choose instance and label manually 
Manual = clean_text("Although better than the sequel, this still sucks!")
encoding_manual = tokenizer([Manual], padding=True, truncation=True, return_tensors='pt', max_length=16)
###########################################################################

# Make prediction and generate explanations
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
sns.heatmap([shap_values_manual[0, :, 0].values], annot=[manual_tokens], fmt="", cmap='coolwarm', cbar=True, xticklabels=manual_tokens, yticklabels=["Explanation Scores"])
plt.xlabel("Tokens")
plt.title(f"SHAP Heatmap for Label {label_idx}")
plt.show()


###########################################################################
