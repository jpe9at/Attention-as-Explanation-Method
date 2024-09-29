import re
import torch
from torch.utils.data import Dataset, DataLoader  # Ensure DataLoader is imported
from transformers import BertTokenizer

def clean_text(text):
    """Cleans the input text by lowering case, removing URLs, numbers, punctuation, and extra spaces."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = text.strip()
    return text

class TextData(Dataset):
    """Dataset class for text data."""
    def __init__(self, texts, labels, tokenizer, max_length=64, train = True):
        self.texts = texts 
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train
    
    def __getitem__(self, index):
        if index >= len(self.texts):
            raise IndexError(f"Index {index} is out of range for dataset length {len(self.texts)}")

        text = self.texts.iloc[index] 
        labels = self.labels[index]

        # Clean the text
        text = clean_text(text)

        # Tokenize and encode the input text
        encoded_input = self.tokenizer(
            text, 
            padding='max_length',  # Pad to max length
            truncation=True,       # Truncate if longer than max_length
            max_length=self.max_length,
            return_tensors='pt'   # Return PyTorch tensors
        )

        input_ids = encoded_input['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoded_input['attention_mask'].squeeze(0)

        # Set requires_grad=False explicitly
        if self.train == False: 
            input_ids.requires_grad = False
            attention_mask.requires_grad = False

        # Return the processed inputs and the label
        return input_ids, attention_mask, labels.to(input_ids.device)  # Ensure labels are on the same device

    def __len__(self):
        return len(self.texts)

class DataModule: 
    """Data module for loading datasets and creating data loaders."""
    def __init__(self, X, y, tokenizer, max_length=164):
        self.dataset = TextData(X, y, tokenizer, max_length)

    def get_dataloader(self, batch_size, num_workers=0):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

