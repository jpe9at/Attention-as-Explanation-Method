import torch
import torch.nn as nn
import time
from CustomDataClass import TextData
import optuna
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score

class Trainer: 
    """The base class for training models with data."""
    def __init__(self, max_epochs=1, batch_size=36, early_stopping_patience=6, min_delta=0.009):
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.num_epochs_no_improve = 0
        self.min_delta = min_delta


    def prepare_val_data(self, text_data):
        self.val_dataloader = DataLoader(text_data, batch_size=self.batch_size, shuffle=True)
    
    def prepare_training_data(self, text_data):
        self.train_dataloader = DataLoader(text_data, batch_size=self.batch_size, shuffle=True)
    
    def prepare_test_data(self, test_dataset):
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
    def prepare_model(self, model):
        model.trainer = self
        self.model = model

    def fit(self, model, train_dataset, val_dataset):
        self.train_loss_values = []
        self.val_loss_values = []
        self.prepare_training_data(train_dataset)
        self.prepare_val_data(val_dataset)
        self.prepare_model(model)
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss, val_loss = self.fit_epoch()

            if (epoch+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train_Loss: {train_loss:.4f}, Val_Loss: {val_loss: .4f}')
            
            self.train_loss_values.append(train_loss)
            self.val_loss_values.append(val_loss)

            #########################################
            #Early Stopping Monitor
            if (self.best_val_loss - val_loss) > self.min_delta:
                self.best_val_loss = val_loss
                self.num_epochs_no_improve = 0
            else:
                self.num_epochs_no_improve += 1
                if self.num_epochs_no_improve == self.early_stopping_patience:
                    print("Early stopping at epoch", epoch)
                    break
            ########################################

            ########################################
            #Scheduler for adaptive learning rate
            #if self.model.scheduler is not None:
            #    self.model.scheduler.step(val_loss)
            ########################################


    def fit_epoch(self):
        train_loss = 0.0
        total_batches = len(self.train_dataloader)

        device = next(self.model.parameters()).device
        for idx, (text, attention_mask, labels) in enumerate(self.train_dataloader):
            # Move inputs to the device
            #text = text.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            output, _ = self.model(text, attention_mask)
            loss = self.model.loss(output, labels)
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            train_loss += loss.item() * text.size(0)

            # Print progress
            time.sleep(0.1)  # Simulate batch processing time

            progress = (idx + 1) / total_batches * 100
            print(f"\rBatch {idx + 1}/{total_batches} completed. Progress: {progress:.2f}%", end='', flush=True)

        train_loss /= len(self.train_dataloader.dataset)
        print(train_loss)
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for text, attention_mask, labels in self.val_dataloader:
                labels = labels.to(device)
                val_output, _ = self.model(text, attention_mask)
                loss = self.model.loss(val_output, labels)
                val_loss += loss.item() * text.size(0) #why multiplication with 0?
            val_loss /= len(self.val_dataloader.dataset)
        return train_loss, val_loss

    def test(self, model, data):
        model.eval()
        self.prepare_test_data(data)
        all_targets = []
        all_predictions = []

        device = next(model.parameters()).device
        
        with torch.no_grad():
            for text, attention_mask, labels in self.test_dataloader:
                # Move inputs to the device
                text = text.to(device)
                attention_mask = attention_mask.to(device)
                
                y_hat, _ = model(text, attention_mask)
                probabilities = torch.sigmoid(y_hat)
                all_targets.append(labels)
                all_predictions.append(probabilities)
        
        all_targets = torch.cat(all_targets).cpu()  # Move to CPU for metrics calculation
        all_predictions = torch.cat(all_predictions).cpu()

        y_true = all_targets.numpy()
        y_pred_prob = all_predictions.numpy() 

        # Metrics calculation
        subset_acc = accuracy_score(y_true, (y_pred_prob > 0.5).astype(int))
        hamming = hamming_loss(y_true, (y_pred_prob > 0.5).astype(int))
        #roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')

        return subset_acc, hamming # , roc_auc
    
    def test_multiclass(self, model, data):
            model.eval()
            self.prepare_test_data(data)
            all_targets = []
            all_predictions = []
            
            device = next(self.model.parameters()).device

            with torch.no_grad():
                for text, attention_mask, labels in self.test_dataloader:
                    # Move inputs to the device
                    text = text.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    y_hat, _ = model(text, attention_mask)
                    predictions = torch.argmax(y_hat, dim = 1) #shape (batch_size, num_classes)
                    all_targets.append(labels)
                    all_predictions.append(predictions)
            
            all_targets = torch.cat(all_targets).cpu()  # Move to CPU for metrics calculation
            all_predictions = torch.cat(all_predictions).cpu()

            y_true = all_targets.numpy()
            y_pred = all_predictions.numpy() 

            # Metrics calculation
            subset_acc = accuracy_score(y_true, y_pred)
            #hamming = hamming_loss(y_true, (y_pred_prob > 0.5).astype(int))
            #roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')

            return subset_acc #, hamming # , roc_auc

