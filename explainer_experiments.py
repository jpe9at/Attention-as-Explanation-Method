
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


# Path filepath as argument.
parser = argparse.ArgumentParser()

# Step 2: Add the file path argument
parser.add_argument('file_path', type=str, help='The path to the input file')
parser.add_argument('path_to_load', type=str, help='The path to load the trained model from')

# Step 3: Parse the arguments
args = parser.parse_args()

path = args.file_path
model_path = args.path_to_load 

######################################################################
#Load the testdata and the model
######################################################################
'''
print('Load the Jigsaw dataframe')
tweet_dataframe = pd.read_csv(path)

tweet_dataframe['comment_text'] = tweet_dataframe['comment_text'].apply(clean_text)

# Split data into features (X) and labels (y)
X = tweet_dataframe['comment_text']
y = tweet_dataframe.iloc[:, 2:]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')

# Create the test dataset
test_dataset = TextData(X_test[-500:], y_test[-500:], tokenizer, max_length=25, train=False)

print('Load the model')
model = torch.load(model_path).to(device)
model.eval()
trainer = Trainer(64)
#trainer.prepare_test_data(test_dataset)

subset_acc, hamming = trainer.test(model,test_dataset)
print(f'Accuracy for base model {subset_acc}')

uniform_attention_model = BERTClassifier('prajjwal1/bert-tiny', 6).to(device)
uniform_attention_model.load_state_dict(model.state_dict(), strict=False)

subset_acc, hamming = trainer.test(uniform_attention_model,test_dataset)
print(f'Accuracy for uniform attention model {subset_acc}')
'''

############################################
'''
print('loading dbpedia')
dataset = load_dataset("dbpedia_14")

test_dataset = pd.DataFrame(dataset["test"])

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
# Create the dataset
test_dataset = TextData(test_dataset['content'][0:1000], test_dataset['label'][0:1000], tokenizer, max_length=25, train=True, labels_datatype = 'long')

model = torch.load('model_db_pedia.pth').to(device)
model.eval()
trainer = Trainer(64)
#trainer.prepare_test_data(test_dataset)

subset_acc = trainer.test_multiclass(model,test_dataset)
print(f'Accuracy for base model {subset_acc}')

uniform_attention_model = BERTClassifier('prajjwal1/bert-tiny', 14, uniform = True).to(device)
uniform_attention_model.load_state_dict(model.state_dict(), strict=False)

subset_acc = trainer.test_multiclass(uniform_attention_model,test_dataset)
print(f'Accuracy for uniform attention model {subset_acc}')
###############################################
'''
print('loading imdb')
dataset = load_dataset("imdb")

test_dataset = pd.DataFrame(dataset["test"][0:50])

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
# Create the dataset
test_dataset = TextData(test_dataset['text'], test_dataset['label'].to_frame(), tokenizer, max_length=25, train=False, labels_datatype = 'long')

model = torch.load('model_imdb.pth').to(device)
model.eval()
trainer = Trainer(64)
#trainer.prepare_test_data(test_dataset)

subset_acc = trainer.test(model,test_dataset)
print(f'Accuracy for base model {subset_acc}')

uniform_attention_model = BERTClassifier('prajjwal1/bert-tiny', 1, uniform = True).to(device)
uniform_attention_model.load_state_dict(model.state_dict(), strict=False)

subset_acc = trainer.test(uniform_attention_model,test_dataset)
print(f'Accuracy for uniform attention model {subset_acc}')
'''
#####################################################################

print('loading yelp')
dataset = load_dataset("yelp_polarity")

test_dataset = pd.DataFrame(dataset["test"][0:50])

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
# Create the dataset
test_dataset = TextData(test_dataset['text'], test_dataset['label'].to_frame(), tokenizer, max_length=25, train=False, labels_datatype = 'long')

model = torch.load('model_yelp.pth').to(device)
model.eval()
trainer = Trainer(64)
#trainer.prepare_test_data(test_dataset)

subset_acc = trainer.test(model,test_dataset)
print(f'Accuracy for base model {subset_acc}')

uniform_attention_model = BERTClassifier('prajjwal1/bert-tiny', 1, uniform = True).to(device)
uniform_attention_model.load_state_dict(model.state_dict(), strict=False)

subset_acc = trainer.test(uniform_attention_model,test_dataset)
print(f'Accuracy for uniform attention model {subset_acc}')





#####################################################################
#Define two helper functions for the experiments
#####################################################################
#Define a model wrapper function for the SHAP explianer
def model_wrapper(sample_inputs):
    sample_encodings = tokenizer(sample_inputs.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=25)
    sample_encodings = {key: tensor.to(device) for key, tensor in sample_encodings.items()}
    output, _ = model(sample_encodings['input_ids'], sample_encodings['attention_mask'])
    return torch.sigmoid(output)

def model_wrapper_2(sample_inputs):
    sample_encodings = tokenizer(sample_inputs.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=25)
    sample_encodings = {key: tensor.to(device) for key, tensor in sample_encodings.items()}
    output, _ = uniform_attention_model(sample_encodings['input_ids'], sample_encodings['attention_mask'])
    return torch.sigmoid(output)


#define a function that calculates correlations between explanation values
def rowwise_correlation(tensor1, tensor2):
    correlations = []
    for row1, row2 in zip(tensor1, tensor2):
       
       # Compute mean of each row covariance and std
        mean1, mean2 = row1.mean(), row2.mean()
        cov = ((row1 - mean1) * (row2 - mean2)).sum() / (len(row1[-1]))
        std1 = row1.std()
        std2 = row2.std()
        # Pearson correlation
        correlation = cov / (std1 * std2)
        if np.isnan(correlation):
            correlation = 0
            correlations.append(correlation)
        else: 
            correlations.append(correlation.item())
    
    return correlations

#define a list of irrelevant tokens

irrelevant_tokens = ['.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', "'", '"', '[CLS]', '[MASK]', '[PAD]', '[SEP]']

vocab = tokenizer.get_vocab()

irrelevant_token_ids = {}
for token in irrelevant_tokens:
    token_id = vocab.get(token)
    if token_id is not None:
        irrelevant_token_ids[token] = token_id

list_of_irrelevant_tokens = list(irrelevant_token_ids.values())

############################################################################
#Start the experiments
#Test for:
    #Ability to reflect the model's prediction
    #Explanatory values of irrelvant tokens
    #Correlation between shap and attention
############################################################################

# Initialize explainer object
def model_experiments(model, wrapper):
    explainer = shap.Explainer(wrapper, tokenizer)

    classifier_output = []
    att_explanations = []
    shap_explanations = []

    times_attention_calculation = []
    times_SHAP_calculation =[]

    class_criterion_list = []

    with torch.no_grad():
        for text, attention_mask, labels in trainer.test_dataloader:
            # Move inputs to the device
            text = text.to(device)
            attention_mask = attention_mask.to(device)
                        
            class_criterion = np.isin(text[:, :], list_of_irrelevant_tokens)[..., np.newaxis]  
            class_criterion_list.append(class_criterion)

            start_time = time.time()
            y_hat, attention_scores = model(text, attention_mask, exp_scores = True)
            #here softmax for multiclass
            probabilities = torch.sigmoid(y_hat)
            #probabilities = torch.softmax(y_hat, dim = 1)
            classifier_output.append(probabilities)

            #get attention_explanation scores
            att_explanations.append(attention_scores) #Shape (batch, label, length_secquence)
            end_time = time.time()
            times_attention_calculation.append(end_time - start_time)

            #get shap_explanation scores
            decoded_sentences = tokenizer.batch_decode(text, skip_special_tokens=True)
            start_time = time.time()
            shap_values = explainer(decoded_sentences) # Shape (batch, length_sequence, label)
            padded_arrays = np.array([np.pad(arr, ((0, 25 - arr.shape[0]), (0, 0)), 'constant') for arr in shap_values.values])
            transposed_arrays = np.transpose(padded_arrays, (0, 2, 1))
            shap_explanations.append(transposed_arrays)
            end_time = time.time()
            times_SHAP_calculation.append(end_time - start_time)

           

    #calculate the explainer based classifier
    classifier_output = torch.cat(classifier_output).numpy()
    classifier_predictions = (classifier_output > 0.5).astype(int)

    att_explanations = torch.cat(att_explanations)
    att_to_all_tokens = torch.sum(att_explanations, dim = 2).numpy()
    att_explanation_classifier = (att_to_all_tokens > 0).astype(int)

    shap_explanations= np.concatenate(shap_explanations, axis = 0)
    shap_to_all_tokens = np.sum(shap_explanations, axis = 2)
    base_line = np.mean(classifier_output, axis=0)  
    print('Baseline of predictions')
    print(base_line)
    shap_prediction = shap_to_all_tokens + base_line 
    shap_explanation_classifier = (shap_prediction > 0.5).astype(int)

    #calculate accuracy of classifiers
    subset_acc_att = accuracy_score(classifier_predictions, att_explanation_classifier)
    hamming_att = hamming_loss(classifier_predictions, att_explanation_classifier)
    total_att_time = sum(times_attention_calculation) 
    print(f'Accuracy of Attention based explanation: {subset_acc_att}')
    print(f'Hamming_loss of Attention based explanation: {hamming_att}')
    print(f'Total time for calculating attention scores: {total_att_time}')


    subset_acc_shap = accuracy_score(classifier_predictions, shap_explanation_classifier)
    hamming_shap = hamming_loss(classifier_predictions, shap_explanation_classifier)
    total_shap_time = sum(times_SHAP_calculation)
    print(f'Accuracy of SHAP explanation: {subset_acc_shap}')
    print(f'Hamming_loss of SHAP explanation: {hamming_shap}')
    print(f'Total time for calculating shap values: {total_shap_time}')


    #claculate relvenace scores for irrelevant tokens
    class_criteria = np.concatenate(class_criterion_list)
    att_to_irrelevant_tokens = torch.sum(att_explanations * np.transpose(class_criteria, (0,2,1)), dim = 2).numpy()
    share_of_irrelevant_attention_tokens= np.mean(att_to_irrelevant_tokens / att_to_all_tokens, axis = 0)

    shap_to_irrelevant_tokens = (np.sum(shap_explanations * np.transpose(class_criteria, (0,2,1)), axis = 2))
    share_of_irrelevant_shap_tokens = np.mean(shap_to_irrelevant_tokens / shap_to_all_tokens, axis = 0)
    print(f'Share of irrelevant tokens for attention scores {share_of_irrelevant_attention_tokens} ')
    print(f'Share of irrelevant tokens for shap scores {share_of_irrelevant_shap_tokens} ')

    #calculate correlations between shap and attention scores
    return att_explanations, shap_explanations 


att_exp, shap_exp = model_experiments(model, model_wrapper)
correlations1 = rowwise_correlation(att_exp.numpy(), shap_exp)
print('k')
att_exp, shap_exp = model_experiments(uniform_attention_model, model_wrapper_2)
correlations2 = rowwise_correlation(att_exp.numpy(), shap_exp)


plt.figure()
plt.boxplot(correlations1)
plt.title("Boxplot of Row-wise Correlations base model")
plt.ylabel("Correlation Value")

plt.figure()
plt.boxplot(correlations2)
plt.title("Boxplot of Row-wise Correlations uniform_attention_model")
plt.ylabel("Correlation Value")


plt.show()
