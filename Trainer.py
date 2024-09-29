import torch
import torch.nn as nn
import time
from CustomDataClass import TextData
import optuna
#from Module import EncoderDecoderMaster
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score

class Trainer: 
    """The base class for training models with data."""
    def __init__(self, max_epochs = 1, batch_size = 36, early_stopping_patience=6, min_delta = 0.09, num_gpus=0, max_length = 16):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        #self.early_stopping_patience = early_stopping_patience
        #self.best_val_loss = float('inf')
        #self.num_epochs_no_improve = 0
        #self.min_delta = min_delta
        #assert num_gpus == 0, 'No GPU support yet'
        self.max_length = max_length

    def prepare_training_data(self, text_data):
        self.train_dataloader = DataLoader(text_data, batch_size=self.batch_size, shuffle=True)
    
    def prepare_test_data(self, test_dataset):
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
            
    def prepare_model(self, model):
        model.trainer = self
        self.model = model
    
    def fit(self, model, dataset):
        self.train_loss_values = []
        self.val_loss_values = []
        self.prepare_training_data(dataset)
        self.prepare_model(model)
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = self.fit_epoch()
            '''
            if (epoch+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train_Loss: {train_loss:.4f}, Val_Loss: {val_loss: .4f}, LR = {self.model.scheduler.get_last_lr() if self.model.scheduler is not None else self.model.learning_rate}')
            self.train_loss_values.append(train_loss)
            self.val_loss_values.append(val_loss)

            #########################################
            #Early Stopping Monitor
            #instead, we can also use the early stopping monitor class below. 
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
            if self.model.scheduler is not None:
                self.model.scheduler.step(val_loss)
            ########################################

'''

    def fit_epoch(self):
        train_loss = 0.0
        total_batches = len(self.train_dataloader)
        #torch.autograd.set_detect_anomaly(True)
        for idx, (text, attention_mask, labels) in enumerate(self.train_dataloader):
            output, _ = self.model(text, attention_mask)
            loss = self.model.loss(output,labels)
            self.model.optimizer.zero_grad()
            loss.backward()

            self.model.optimizer.step()
            train_loss += loss.item() * text.size(0)
            
            time.sleep(0.1)  # Simulate batch processing time
            
            # Calculate progress
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
            for  text, attention_mask, labels  in self.test_dataloader:
                y_hat, _ = model(text, attention_mask)
                probabilities = torch.sigmoid(y_hat)
                all_targets.append(labels)
                all_predictions.append(probabilities)
                print('batch_done') 
        all_targets = torch.cat(all_targets)
        all_predictions = torch.cat(all_predictions)

        y_true = all_targets.numpy()
        y_pred_prob = all_predictions.numpy() 

        # Metrics calculation
        # Subset Accuracy (Exact Match Ratio)
        subset_acc = accuracy_score(y_true, (y_pred_prob > 0.5).astype(int))  # Example using threshold here for accuracy
        hamming = hamming_loss(y_true, (y_pred_prob > 0.5).astype(int))
    
        # For continuous probability metrics
        # Calculate ROC AUC for each class (macro-average)
        roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')
 
        return subset_acc, hamming, roc_auc



    
    @classmethod
    def Optuna_objective(cls, trial, workload_data, input_size, output_size):
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        learning_r = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32,128,256])
        hidden_size = trial.suggest_categorical('hidden_size',[32,64,128])
        hidden_size_2 = trial.suggest_categorical('hidden_size_2',[8,32,64])
        l2_rate = trial.suggest_categorical('l2_rate', [0.0,0.0001,0.005])
        loss_function = trial.suggest_categorical('loss', ['MSE','Huber'])
        window_size = trial.suggest_categorical('window_size', [10,15,20])
        gradient_clip = trial.suggest_categorical('gradient_clip', [0.0,1.0])
        scheduler = trial.suggest_categorical('scheduler', [None, 'OnPlateau'])
        num_layers = trial.suggest_categorical('num_layers', [1])

        model = EncoderDecoderMaster(input_size, hidden_size, 3, hidden_size_2, output_size, num_layers, learning_rate = learning_r, loss_function = loss_function, clip_val = gradient_clip, scheduler = scheduler)
        trainer = cls(30,  batch_size, window_size = window_size)
        trainer.fit(model, workload_data)

        return  trainer.val_loss_values[-1]

    @classmethod
    def hyperparameter_optimization(cls, workload_data, input_size,output_size):
        study = optuna.create_study(direction='minimize')
        objective_func = lambda trial: cls.Optuna_objective(trial, workload_data, input_size, output_size)
        study.optimize(objective_func, n_trials=15)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_accuracy = best_trial.value

        return best_params, best_accuracy
'''
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for text, attention_mask, labels in self.val_dataloader:
                additional = y_batch[:,-1,3:]
                val_output = self.model(x_batch, additional)
                loss = self.model.loss(val_output, y_batch[:,-1,0:3])
                val_loss += loss.item() * x_batch.size(0) #why multiplication with 0?
            val_loss /= len(self.val_dataloader.dataset)
            
        return train_loss #, val_loss
'''

