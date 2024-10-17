#!/usr/bin/env python

'''
run LSTM models on resting state of ukb as a pretrained model
'''

#import pandas as pd
import glob, sys
import numpy as np
import pandas as pd
import os

from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
from nilearn.maskers import NiftiMasker,NiftiLabelsMasker
from sklearn import metrics
import warnings
import copy
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import re
import math
import time
import glob
import random
import string
import collections

import wandb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupShuffleSplit

from tqdm import tqdm

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
torch.cuda.is_available()




def Check2Runs(s, data_dir):
    run1 = glob.glob(data_dir + f'sub-{s}_run-1*')
    run2 = glob.glob(data_dir + f'sub-{s}_run-2*')
    
    if run1 and run2:
        return True
    else:
        return False
    
def FormatSubID(source_id):
    # Make it a string
    source_id = str(source_id)
    if "bio" in source_id:   
        # check if the sub number is four digit:
        sub_num = int(source_id.split("bio")[1])
        formatted_id = "bio" + "{:04}".format(sub_num)
        
    elif "Bio" in source_id:
        # check if the sub number is four digit:
        sub_num = int(source_id.split("Bio")[1])
        formatted_id = "bio" + "{:04}".format(sub_num)
        
    elif type(int(source_id))==int: #sub_id just numbers
        sub_num = int(source_id)
        formatted_id = "bio" + "{:04}".format(sub_num)
    
    else: # everything else just return as it is..
        formatted_id = source_id
    
    return formatted_id

class CustomRSFCMDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y
    
def ParcelFileToTensor(parcel_file_list, NROI, rs_timepoint = 750, transform = None, network = None):
    """
    Convert pregenerated parcellation files to tensor

    Args:
    parcel_file_list: list
        list of parcellation files to extract the data
    network: str 
        which network to lesion, or default is no lesion 
        the network name must be in the dataframe column names

    Returns:
    X_tensor: Tensor
    """
    
    X = []

    for parcellation_path in parcel_file_list:
        par_df =pd.read_csv(parcellation_path)
        
        if par_df.shape[0] != rs_timepoint:
            raise ValueError(f'Error: {parcellation_path} has {par_df.shape[0]} time points, not {rs_timepoint}')
        else: 
            # Lesion first:
            if network:
                x_cols = [col for col in par_df.columns if 'Networks' in col]
                zero_cols = [col for col in x_cols if (network in col)]
                par_df[zero_cols]=0
                
            this_x = par_df.iloc[:,1:NROI+1].astype(float).values
            # Transform:
            if transform:
                this_x = transform.fit_transform([this_x])[0]
                
            X.append(this_x)   
        # Convert to tensor
        # sub x time point x NROI
        X_tensor = torch.tensor(np.array(X))
   
    return X_tensor


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
class LSTMWithTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, num_heads, num_layers, dropout=0.2, dropout_layer=0.2):
        super(LSTMWithTransformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.dropout_layer = nn.Dropout(dropout_layer)
   
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pass through the transformer encoder
        transformer_out = self.transformer(lstm_out)
        
        # Apply global average pooling
        context_vector = torch.mean(transformer_out, dim=1)
        context_vector = self.dropout_layer(context_vector)
        
        # Output layer
        output = self.fc(context_vector)
        return output
    
class BiLSTMWithTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, num_heads, num_layers, dropout=0.2, dropout_layer=0.2):
        super(BiLSTMWithTransformer, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim*2, nhead=num_heads),
            num_layers=num_layers
        )
        self.dropout_layer = nn.Dropout(dropout_layer)
   
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        bilstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Pass through the transformer encoder
        transformer_out = self.transformer(bilstm_out)
        
        # Apply global average pooling
        context_vector = torch.mean(transformer_out, dim=1)
        context_vector = self.dropout_layer(context_vector)
    
        # Output layer
        output = self.fc(context_vector)
        return output
    
        
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, dropout=0.2, dropout_layer=0.2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout_layer)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        context_vector = self.dropout_layer(context_vector)

        output = self.fc(context_vector)
        return output
    
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, dropout=0.2, dropout_layer=0.2):
        super(BiLSTMWithAttention, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout_layer = nn.Dropout(dropout_layer)

    def forward(self, x):
        bilstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(bilstm_out), dim=1)
        context_vector = torch.sum(attention_weights * bilstm_out, dim=1)
        context_vector = self.dropout_layer(context_vector)
        
        output = self.fc(context_vector)
        return output    

class LSTN(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, output_size, dropout=0.2):
        super(LSTN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        # out = nn.functional.softmax(out, dim=1)  # Apply softmax activation function
        return out


class BLSTN(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, output_size, dropout=0.2):
        super(BLSTN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden_size by 2 for bidirectional LSTM
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(torch.cat((h[-2], h[-1]), dim=1))  # Concatenate the hidden states from both directions
        return out
    
    
def log_my_plot(y_test, y_pred, y_pred_sigmoid):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    # confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='BuPu', fmt='d', xticklabels=['Low PainIn', 'High PainIn'], yticklabels=['Low PainIn', 'High PainIn'],annot_kws={"size": 16})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    wandb.log({"confusion_matrix_plot": wandb.Image(plt)})
    plt.close()

    # calculate accuracy:
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy:.4f}')

    print('chance level: ', 1 - np.mean(y_test))
    # calculate sensitivity and specificity
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')

    # Compute the ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_sigmoid)
    roc_auc = auc(fpr, tpr)

    # Create the AUC plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Log the AUC plot to WandB
    wandb.log({"AUC_plot": wandb.Image(plt)})
    plt.close()

def get_model_dir(run_name, root_path):
    import os
    os.makedirs(os.path.join(root_path, 'saved_models', run_name), exist_ok=True)
    return os.path.join(root_path, 'saved_models', run_name)

def count_params(model):
    """
    Print the total number of parameters in the model.

    Parameters:
    - model: PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def create_labels_df(data_set, data_dir, ROI_scheme, project_dir='/home/yiyuw/projects/HEAL', create=True):
    if data_set == 'HEAL':
        print(f'using ROI scheme - {ROI_scheme}')
        # device = rank #device = torch.device(f"cuda:{rank}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warnings.filterwarnings('ignore')

        label_df = pd.read_csv(f'{project_dir}/labels.csv')

        demographics = pd.read_csv(f'{project_dir}/participants_demographics.csv')

        label_df = pd.read_csv(f'{project_dir}/labels.csv')
        label_df = label_df[label_df.session == 'baseline']
        CP_df = pd.merge(label_df, demographics, on='subject_id')
        # include rest run 1 and run 2
        augmented_df = pd.DataFrame(columns=CP_df.columns.to_list() + ['run', 'file'])
        for f in np.sort(glob.glob(f'{data_dir}/{ROI_scheme}/*task-rest*')):
            if pd.read_csv(f).shape[0] != 750:
                print(f'Error: {f} has {pd.read_csv(f).shape[0]} time points, not 750. Skipping...')
                continue
            else:
                s = re.search(r'sub-(\w+)_task-rest_run-(\d+)_parcellation', f).group(1)
                run = re.search(r'sub-(\w+)_task-rest_run-(\d+)_parcellation', f).group(2)
                df = CP_df[CP_df['subject_id']==s]
                df['run'] = run
                df['file'] = f
                augmented_df = pd.concat([augmented_df, df])
        # choose run or runs:
        augmented_df = augmented_df.loc[augmented_df['run'].isin(['1', '2'])]        
        augmented_df.reset_index(drop=True, inplace=True)
        augmented_df.to_csv(f'{project_dir}/HEAL_mixed_augmented_df.csv', index=False)
    elif data_set == 'UKB':
        if create:  
            print('making new labels file for ukb')
            augmented_df = pd.read_csv(f'{project_dir}/ukb_labels/ukb_labels.csv')
            augmented_df['file'] = None
            for s in augmented_df['eid']:
                file_path = f"{data_dir}/{ROI_scheme}/sub-{s}_task-rest_parcellation-{ROI_scheme}.csv"
                augmented_df.loc[augmented_df['eid']==s, 'file'] = file_path
                # if there is no nifti file, remove the row (i.e., this subject)
                if not os.path.exists(file_path):
                    print(f'removing {s} from the dataframe')
                    augmented_df = augmented_df[augmented_df['eid']!=s]
            augmented_df.reset_index(drop=True, inplace=True)
            # rename eid column to subject_id
            augmented_df.rename(columns={'eid':'subject_id'}, inplace=True)
            # save to csv
            augmented_df.to_csv(f'{project_dir}/ukb_labels/ukb_labels_filecheck.csv', index=False)
        else:
            augmented_df = pd.read_csv(f'{project_dir}/ukb_labels/ukb_labels_filecheck.csv')
    return augmented_df



class MetaLearner:
    def __init__(self, model, lr_inner=0.01, lr_meta=0.001, save_model_dir=None):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_meta = lr_meta
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.lr_meta)
        self.save_model_dir = save_model_dir

    def inner_update(self, task_loader, num_inner_steps=1):
        """ Perform task-specific training in the inner loop. """
        task_model = copy.deepcopy(self.model)
        task_model.train()

        optimizer = optim.SGD(task_model.parameters(), lr=self.lr_inner)
        y_preds_list, y_preds_prob_list, y_train_list = [], [], []
        step_loss = 0
        for step in range(num_inner_steps):
            for batch_data, batch_labels in tqdm(task_loader, desc=f"Inner Loop Step {step+1}/{num_inner_steps}"):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                preds = task_model(batch_data.float())
                loss = nn.CrossEntropyLoss()(preds, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_loss += loss.item() * batch_data.size(0)
                probas_class_1 = F.softmax(preds, dim=1)[:, 1]  # Extract probability for class 1
                y_preds_list.append(torch.argmax(preds, dim=1).detach().cpu().numpy())
                y_preds_prob_list.append(probas_class_1.detach().cpu().numpy())
                y_train_list.append(batch_labels.detach().cpu().numpy())

            # calculate step auc and accuracy
            y_preds_list_flattened = [item for sublist in y_preds_list for item in sublist]
            y_preds_prob_list_flattened = [item for sublist in y_preds_prob_list for item in sublist]
            y_train_list_flattened = [item for sublist in y_train_list for item in sublist]
            
            step_auc = roc_auc_score(y_train_list_flattened, y_preds_prob_list_flattened)
            step_accuracy = accuracy_score(y_train_list_flattened, y_preds_list_flattened)
            print(f"Inner Loop Step {step+1}/{num_inner_steps}: Loss: {loss.item():.4f}, Accuracy: {step_accuracy:.4f}, AUC: {step_auc:.4f}")
            
            step_loss /= len(task_loader)
            wandb.log({
                f"inner_loss_step": loss.item(),
                f"inner_accuracy": step_accuracy,
                f"inner_auc_step": step_auc,
                f"inner_loss_step": step_loss
            })
        return task_model

    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate accuracy and AUC for binary classification."""
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        return accuracy, auc

    def meta_update(self, task_models, task_val_loaders, meta_epoch):
        """ Perform the meta-update based on task validation performance. """
        self.meta_optimizer.zero_grad()

        meta_loss = 0
        total_accuracy = 0
        total_auc = 0
        num_samples = 0

        for task_model, task_val_loader in zip(task_models, task_val_loaders):
            task_model.train()  # Ensure it's in training mode to allow gradients

            all_preds = []
            all_labels = []
            all_probas = []

            for val_data, val_labels in tqdm(task_val_loader, desc="Validation DataLoader"):
                val_data = val_data.to(device)
                val_labels = val_labels.to(device)
                preds = task_model(val_data.float())
                loss = nn.CrossEntropyLoss()(preds, val_labels)

                # Accumulate validation loss
                meta_loss += loss
                probas_class_1 = F.softmax(preds, dim=1)[:, 1]  # Extract probability for class 1
                # Store predictions and true labels for accuracy/AUC calculation
                all_labels.extend(val_labels.cpu().numpy())
                all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())  # Binary class predictions
                all_probas.extend(probas_class_1.detach().cpu().numpy())  # Assuming class 1's probability

            # Calculate metrics
            accuracy, auc = self.calculate_metrics(all_labels, all_preds, all_probas)
            total_accuracy += accuracy * len(all_labels)
            total_auc += auc * len(all_labels)
            num_samples += len(all_labels)

        # Average the loss and metrics across tasks
        meta_loss /= len(task_models)
        avg_accuracy = total_accuracy / num_samples
        avg_auc = total_auc / num_samples

        # Backpropagate through the meta-loss
        meta_loss.backward()  # Perform backpropagation
        self.meta_optimizer.step()

        # Log the meta-loss, accuracy, and AUC to WandB
        wandb.log({
            "meta_loss": meta_loss.item(),
            "meta_accuracy": avg_accuracy,
            "meta_auc": avg_auc
        })

        print(f"Meta Loss: {meta_loss.item():.4f}, Accuracy: {avg_accuracy:.4f}, AUC: {avg_auc:.4f}")

    def train_on_tasks(self, task_train_loaders, task_val_loaders, target_test_loader, num_meta_epochs=10, num_inner_steps=1):
        """ Train the model using meta-learning across tasks (datasets). """
        for epoch in range(num_meta_epochs):
            task_models = []

            # Inner loop: Adaptation on each task (dataset)
            for task_loader in task_train_loaders:
                task_model = self.inner_update(task_loader, num_inner_steps=num_inner_steps)
                task_models.append(task_model)

            # Outer loop: Meta-update using validation data from each task
            self.meta_update(task_models, task_val_loaders, meta_epoch=epoch+1)

            print(f"Meta-Epoch {epoch+1}/{num_meta_epochs} completed.")

            # save the model
            if epoch+1 % 5 == 0:
                torch.save(self.model.state_dict(), f"meta_model_{epoch+1}.pt")

            # after outer loop, evaluate the model on the target test set
            self.evaluate_test_set(task_model, target_test_loader, epoch)


    def fine_tune_on_target_task(self, target_loader, num_fine_tune_steps=5):
        """ Fine-tune the meta-learned model on a target task (Task 1 or Task 2). """
        target_model = copy.deepcopy(self.model)
        optimizer = optim.Adam(target_model.parameters(), lr=self.lr_inner)
        
        # Set the model to training mode
        target_model.train()
        y_preds_list, y_preds_prob_list, y_train_list = [], [], []
        epoch_loss = 0
        for step in range(num_fine_tune_steps):

            for batch_data, batch_labels in tqdm(target_loader):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                # Forward pass
                preds = target_model(batch_data.float())
                loss = nn.CrossEntropyLoss()(preds, batch_labels)

                # Perform backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_data.size(0)
                
                probas_class_1 = F.softmax(preds, dim=1)[:, 1]  # Extract probability for class 1
                y_preds_list.append(torch.argmax(preds, dim=1).detach().cpu().numpy())
                y_preds_prob_list.append(probas_class_1.detach().cpu().numpy())
                y_train_list.append(batch_labels.detach().cpu().numpy())

            y_preds_list_flattened = [item for sublist in y_preds_list for item in sublist]
            y_preds_prob_list_flattened = [item for sublist in y_preds_prob_list for item in sublist]
            y_train_list_flattened = [item for sublist in y_train_list for item in sublist]

            epoch_auc = roc_auc_score(y_train_list_flattened, y_preds_prob_list_flattened)
            epoch_accuracy = accuracy_score(y_train_list_flattened, y_preds_list_flattened)
            epoch_loss /= len(target_loader)

            wandb.log({
                "fine_tune_loss": epoch_loss,
                "fine_tune_accuracy": epoch_accuracy,
                "fine_tune_auc": epoch_auc
            })
            
            print(f"Fine-Tune Step {step+1}/{num_fine_tune_steps}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, AUC: {epoch_auc:.4f}")
        return target_model

    def evaluate_test_set(self, task_model, test_loader, epoch):
        """ Evaluate the meta learning model on the target task's test set using DataLoader. """
        model = copy.deepcopy(task_model).eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for batch_data, batch_labels in tqdm(test_loader):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                # Forward pass
                preds = model(batch_data.float())
                loss = nn.CrossEntropyLoss()(preds, batch_labels)
                
                # Accumulate loss
                total_loss += loss.item()

                # Compute accuracy
                predicted_labels = torch.argmax(preds, dim=1)
                correct += (predicted_labels == batch_labels).sum().item()
                total += batch_labels.size(0)
                probas_class_1 = F.softmax(preds, dim=1)[:, 1]

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total * 100  # Accuracy in percentage
        AUC = roc_auc_score(batch_labels.detach().cpu().numpy(), probas_class_1.detach().cpu().numpy())
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        print(f"ACU score: {AUC:.4f}")
        wandb.log({'test_loss': avg_loss, 'test_accuracy': accuracy, 'test_auc': AUC})
        
        # save the test_df for visualization
        test_df = pd.DataFrame({'y_true': batch_labels.detach().cpu().numpy(), 'y_preds': predicted_labels.detach().cpu().numpy(), 'y_preds_prob': probas_class_1.detach().cpu().numpy()})
        test_df.to_csv(self.save_model_dir + f'/test_df_epoch-{epoch}.csv', index=False)


        return avg_loss, accuracy, AUC


def evaluate_pretrained_model(model, test_loader, device):
    """ Evaluate the fine-tuned model on the target task's test set using DataLoader. """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_data, batch_labels in tqdm(test_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            # Forward pass
            preds = model(batch_data.float())
            loss = nn.CrossEntropyLoss()(preds, batch_labels)
            
            # Accumulate loss
            total_loss += loss.item()

            # Compute accuracy
            predicted_labels = torch.argmax(preds, dim=1)
            correct += (predicted_labels == batch_labels).sum().item()
            total += batch_labels.size(0)
            probas_class_1 = F.softmax(preds, dim=1)[:, 1]


    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total 
    AUC = roc_auc_score(batch_labels.detach().cpu().numpy(), probas_class_1.detach().cpu().numpy())
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    print(f"ACU score: {AUC:.4f}")
    wandb.log({'test_loss': avg_loss, 'test_accuracy': accuracy, 'test_auc': AUC})

    # the y_preds and y_true for the visualization later

        
    return batch_labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy(), probas_class_1.detach().cpu().numpy(), avg_loss, accuracy, AUC

# Initialize two models for two tasks (can have different number of classes)
num_lstm_layers = 1
hidden_size = 256
batch_size = 16
model_name = 'LSTMWithTransformer'
ROI_scheme = 'schaefer200'
NROI = int(ROI_scheme.split('schaefer')[1])
print(NROI)
lr=0.0001
source_y_variable='sex'
target_y_variable='group'
dropout=0.5
dropout_layer=0.2
num_heads=8
num_layers=2

SEEDS = [30, 35, 42, 50, 60, 70, 80, 90, 100, 88]

for SEED in SEEDS:
    input_size = NROI
    output_size = 2
    # set initialization
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model_name == 'LSTM':
        model = LSTN(input_size, hidden_size, num_lstm_layers, output_size, dropout=dropout)
    elif model_name == 'BLSTM':
        model = BLSTN(input_size, hidden_size, num_lstm_layers, output_size, dropout=dropout)
    elif model_name == 'LSTMWithAttention':
        model = LSTMWithAttention(input_size, hidden_size, num_lstm_layers, output_size, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'BiLSTMWithAttention':

        model = BiLSTMWithAttention(input_size, hidden_size, num_lstm_layers, output_size, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'LSTMWithTransformer':
        model = LSTMWithTransformer(input_size, hidden_size, num_lstm_layers, output_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'BiLSTMWithTransformer':
        model = BiLSTMWithTransformer(input_size, hidden_size, num_lstm_layers, output_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout, dropout_layer=dropout_layer)    
    else:
        raise ValueError("model_name must be: 'LSTM', 'BLSTM', 'LSTMWithAttention', 'BiLSTMWithAttention', 'LSTMWithTransformer', 'BiLSTMWithTransformer' ")    


    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    count_params(model)

    # # Create a MetaLearner for meta-learning across tasks
    # meta_learner = MetaLearner(model=model_task1, lr_inner=0.0001, lr_meta=0.0001)


    # create a mixed_dataset
    project_dir = '/home/yiyuw/projects/HEAL'
    source_data_set = 'UKB'
    target_data_set = 'HEAL'


    target_data_dir = f'{project_dir}/parcellations'
    target_rs_timepioint = 750

    source_data_dir = f'{project_dir}/ukb_parcellations'
    source_rs_timepioint = 490

    mask_img = f'{project_dir}/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'
    source_augmented_df = create_labels_df(source_data_set, source_data_dir, ROI_scheme)
    target_augmented_df = create_labels_df(target_data_set, target_data_dir, ROI_scheme)

    # prep source data
    print("preparing source data")
    source_X, source_y = [], []
    for i, row in source_augmented_df.iterrows():
        f = source_augmented_df.loc[i, 'file']
        source_X.append(f)
        source_y.append(source_augmented_df.loc[i, source_y_variable])
    source_y = np.array(source_y)
    source_subjects_list = np.unique(source_augmented_df['subject_id'].values).tolist()
    source_train_sub_idx, source_val_sub_idx = train_test_split(source_subjects_list, test_size=0.2, random_state=SEED)
    # find the indices of the subjects in the augmented_df
    source_train_idx = source_augmented_df[source_augmented_df['subject_id'].isin(source_train_sub_idx)].index
    source_val_idx = source_augmented_df[source_augmented_df['subject_id'].isin(source_val_sub_idx)].index

    # get the parcel file names per subject
    source_X_train_sub = np.array(source_X)[source_train_idx.astype(int)]
    source_X_val_sub = np.array(source_X)[source_val_idx.astype(int)]

    # load ukb x and y for training
    source_X_train = ParcelFileToTensor(source_X_train_sub, NROI, rs_timepoint=source_rs_timepioint)
    source_y_train = torch.tensor(source_y[source_train_idx.astype(int)])

    # Meta-data for final evaluation step (could be a validation set across tasks)
    source_X_val = ParcelFileToTensor(source_X_val_sub, NROI, rs_timepoint=source_rs_timepioint)
    source_y_val = torch.tensor(source_y[source_val_idx.astype(int)])

    source_train_dataset = CustomRSFCMDataset(source_X_train, source_y_train, transform=None, target_transform=None)
    source_val_dataset = CustomRSFCMDataset(source_X_val, source_y_val, transform=None, target_transform=None)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
    source_val_dataloader = DataLoader(source_val_dataset, batch_size=len(source_y_val), shuffle=False)

    print(source_X_train.shape, source_y_train.shape)
    print(source_X_val.shape, source_y_val.shape)

    # test_dataloader - target_test_dataloader
    # task_train_loaders, task_val_loaders - source_train_dataloader, source_val_dataloader
    # target_loader - target_train_dataloader

    # make target data
    # load HEAL x and y for training

    # make X
    print("preparing target data")
    target_X, target_y = [], []
    for i, row in target_augmented_df.iterrows():
        f = target_augmented_df.loc[i, 'file']
        target_X.append(f)
        target_y.append(target_augmented_df.loc[i, target_y_variable])
    target_y = np.array(target_y)

    # split based on participants for train, test, and val
    target_subjects_list = np.unique(target_augmented_df['subject_id'].values).tolist()
    temp_sub_idx, target_test_sub_idx = train_test_split(target_subjects_list, test_size=0.2, random_state=SEED) #its the same testing subjects across all models
    target_train_sub_idx, target_val_sub_idx = train_test_split(temp_sub_idx, test_size=0.5, random_state=SEED)

    # find the indices of the subjects in the augmented_df
    target_train_idx = target_augmented_df[target_augmented_df['subject_id'].isin(target_train_sub_idx)].index # target meta train data
    target_test_idx = target_augmented_df[target_augmented_df['subject_id'].isin(target_test_sub_idx)].index # target fine-tuning data
    target_val_idx = target_augmented_df[target_augmented_df['subject_id'].isin(target_val_sub_idx)].index # target meta val data

    # get the parcel file names per subject
    target_X_train_sub = np.array(target_X)[target_train_idx.astype(int)]
    target_X_test_sub = np.array(target_X)[target_test_idx.astype(int)]
    target_X_val_sub = np.array(target_X)[target_val_idx.astype(int)]

    # # make X_train and X_test
    target_X_train = ParcelFileToTensor(target_X_train_sub, NROI, rs_timepoint=target_rs_timepioint)
    target_X_test = ParcelFileToTensor(target_X_test_sub, NROI, rs_timepoint=target_rs_timepioint)
    target_X_val = ParcelFileToTensor(target_X_val_sub, NROI, rs_timepoint=target_rs_timepioint)

    target_y_train = torch.tensor(target_y[target_train_idx.astype(int)])
    target_y_test = torch.tensor(target_y[target_test_idx.astype(int)])
    target_y_val = torch.tensor(target_y[target_val_idx.astype(int)])

    print(target_X_train.shape, target_y_train.shape)
    print(target_X_test.shape, target_y_test.shape)
    print(target_X_val.shape, target_y_val.shape)
    # datasets:
    target_train_dataset = CustomRSFCMDataset(target_X_train, target_y_train, transform=None, target_transform=None)
    target_val_dataset = CustomRSFCMDataset(target_X_val, target_y_val, transform=None, target_transform=None)
    target_test_dataset = CustomRSFCMDataset(target_X_test, target_y_test, transform=None, target_transform=None)

    # data loader
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_test_dataloader = DataLoader(target_test_dataset, batch_size=len(target_y_test), shuffle=False)
    target_val_dataloader = DataLoader(target_val_dataset, batch_size=len(target_y_val), shuffle=True)


    # create a mixed dataset for meta learner: 
    task_datasets = [ 
        (source_X_train, source_y_train),  
        (target_X_train, target_y_train)
    ]

    # initialize wandb
    import wandb
    lr_inner = 0.0001
    lr_meta = 0.0001
    num_meta_epochs = 15
    num_inner_steps = 5
    num_fine_tune_steps = 10
    # Initialize WandB run
    config_params = {'num_lstm_layers': num_lstm_layers, 'hidden_size': hidden_size, 'batch_size': batch_size, 
    'model_name': model_name, 
    'ROI_scheme': ROI_scheme, 
    'source_y_variable': source_y_variable, 'target_y_variable': target_y_variable,
    'num_heads': num_heads, 'num_layers': num_layers, 
    'dropout': dropout, 'dropout_layer': dropout_layer, 
    'SEED': SEED, 'source_data_set': source_data_set, 'target_data_set': target_data_set,
    'num_meta_epochs': num_meta_epochs, 'num_inner_steps': num_inner_steps, 'num_fine_tune_steps': num_fine_tune_steps,
    'lr_inner': lr_inner, 'lr_meta': lr_meta}

    project_name = "metalearning_UKB_HEAL_v1"
    run = wandb.init(config=config_params,
        project=project_name,
        entity='yiyuwang1')
    print(f'run_name: {run.name}')
    save_model_dir = get_model_dir(f"{project_name}/" + run.name, project_dir)
    print(f'save_model_dir: {save_model_dir}')
    wandb.init(project=project_name, config=config_params)

    #initial evaluation without trainig
    print("initial evaluation without training")
    evaluate_pretrained_model(model, target_test_dataloader, device)


    meta_learner = MetaLearner(model, lr_inner=lr_inner, lr_meta=lr_meta, save_model_dir=save_model_dir)
    meta_learner.train_on_tasks([source_train_dataloader, target_train_dataloader], 
                                [source_val_dataloader, target_val_dataloader], 
                                target_test_loader=target_test_dataloader,
                                num_meta_epochs=num_meta_epochs, num_inner_steps=num_inner_steps)

    # fine tune the meta-learned model on the target task
    fine_tuned_model = meta_learner.fine_tune_on_target_task(target_train_dataloader, num_fine_tune_steps=num_fine_tune_steps)
    # save the fine-tuned model
    torch.save(fine_tuned_model.state_dict(), f'{save_model_dir}/fine_tuned_model.pth')
    # Evaluate the fine-tuned model on the target task test set
    print("evaluate after fine tuning")
    y_true, y_preds, y_preds_sigmoid, _, _, _ = evaluate_pretrained_model(fine_tuned_model, target_test_dataloader, device)

    test_df = pd.DataFrame({'y_true': y_true, 'y_preds': y_preds, 'y_preds_prob': y_preds_sigmoid})
    test_df.to_csv(f'{save_model_dir}/test_df_after_fine_tuning.csv', index=False)

    # log the model
    wandb.finish()