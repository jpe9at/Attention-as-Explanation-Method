
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


# Path filepath as argument.
parser = argparse.ArgumentParser()

# Step 2: Add the file path argument
parser.add_argument('file_path', type=str, help='The path to the input file')

# Step 3: Parse the arguments
args = parser.parse_args()

# Use the argument in your script
path = args.file_path

print('Load the dataframe')
tweet_dataframe = pd.read_csv(path)

tweet_dataframe['comment_text'] = tweet_dataframe['comment_text'].apply(clean_text)

# Split data into features (X) and labels (y)
X = tweet_dataframe['comment_text']
y = tweet_dataframe.iloc[:, 2:]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Create the dataset
train_dataset = TextData(X_train, y_train, tokenizer, max_length=16, train=True)

# Move the model to the device
model = BERTClassifier('bert-base-uncased', 6).to(device)

# Train the model
trainer = Trainer()
trainer.fit(model, train_dataset)

# Test explanations
# Choose instance and label
item = train_dataset.__getitem__(1)
input_ids, attention_mask, labels = item[0].to(device), item[1].to(device), item[2].to(device)

# Make prediction with explanations
prediction, explanation_scores = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), True)
label_idx = 0

# Plot Attention-based explanation scores
explanation_scores_label = explanation_scores[0, label_idx, :].detach().cpu().numpy()  # Shape: (seq_len,)
tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu())

# Plot the Heatmap for Explanation Scores
plt.figure(figsize=(10, 2))
sns.heatmap([explanation_scores_label], annot=[tokens], fmt="", cmap='coolwarm', cbar=True, xticklabels=tokens, yticklabels=["Explanation Scores"])
plt.xlabel("Tokens")
plt.title(f"Explanation Heatmap for Label {label_idx}")

# Plot SHAP values
model.eval()

# Define a model wrapper function for SHAP
def model_wrapper(sample_inputs):
    sample_encodings = tokenizer(sample_inputs.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
    sample_encodings = {key: tensor.to(device) for key, tensor in sample_encodings.items()}
    output, _ = model(sample_encodings['input_ids'], sample_encodings['attention_mask'])
    return output

# Initialize explainer object
explainer = shap.Explainer(model_wrapper, tokenizer)

# Get SHAP values
decoded_sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
shap_values = explainer([decoded_sentence])  # Shape (batch, length_sequence, label)

# Plot the Heatmap for SHAP Values
plt.figure(figsize=(10, 2))
sns.heatmap([shap_values[0, :, 0].values], annot=[tokens], fmt="", cmap='coolwarm', cbar=True, xticklabels=tokens, yticklabels=["Explanation Scores"])
plt.xlabel("Tokens")
plt.title(f"SHAP Heatmap for Label {label_idx}")
plt.show()
