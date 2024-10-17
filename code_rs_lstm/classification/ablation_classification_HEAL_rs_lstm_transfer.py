#!/usr/bin/env python

'''
run LSTM models on resting state using pretrained model on ukb
'''

#import pandas as pd
import glob, sys
import numpy as np
import pandas as pd
import os

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
    
def ParcelFileToTensor(parcel_file_list, NROI, rs_timepoint= 750, transform = None, network = None):
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
            raise ValueError(f'Error: {parcellation_path} has {par_df.shape[0]} time points, not {rs_timepoint} as specified')
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.2, dropout_layer=0.2):
        super(LSTMWithTransformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.2, dropout_layer=0.2):
        super(BiLSTMWithTransformer, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
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
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, dropout_layer=0.2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
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
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, dropout_layer=0.2):
        super(BiLSTMWithAttention, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
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
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(LSTN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        # out = nn.functional.softmax(out, dim=1)  # Apply softmax activation function
        return out


class BLSTN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(BLSTN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden_size by 2 for bidirectional LSTM
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(torch.cat((h[-2], h[-1]), dim=1))  # Concatenate the hidden states from both directions
        return out
    

def ParceModelParamsFromPretrainedModel(model_dict_path):
    model_name = model_dict_path.split('model-')[1].split('_')[0]
    ROI_scheme = model_dict_path.split('ROI-')[1].split('_')[0]
    hidden_size = int(model_dict_path.split('hidden-')[1].split('_')[0])
    batch_size = int(model_dict_path.split('batch-')[1].split('_')[0])
    lr = float(model_dict_path.split('lr-')[1].split('_')[0])
    dropout = float(model_dict_path.split('dropout-')[1].split('_')[0])
    dropout_layer = float(model_dict_path.split('dropoutlayer-')[1].split('_')[0])
    SEED = int(model_dict_path.split('seed-')[1].split('.')[0])
    num_heads = int(model_dict_path.split('heads-')[1].split('_')[0])
    num_layers = int(model_dict_path.split('layers-')[1].split('_')[0])

    print(f'model_name: {model_name}, ROI_scheme: {ROI_scheme}, hidden_size: {hidden_size}, batch_size: {batch_size}, lr: {lr}, dropout: {dropout}, dropout_layer: {dropout_layer}, SEED: {SEED}, num_heads: {num_heads}, num_layers: {num_layers}')

    return model_name, ROI_scheme, hidden_size, batch_size, lr, dropout, dropout_layer, SEED, num_heads, num_layers


def log_my_plot(y_test, y_pred, Tscore, y_pred_sigmoid):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    # log final results in plots
    # Create a correlation plot using seaborn
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=Tscore, y=y_pred_sigmoid)
    plt.title('Correlation Plot')
    plt.xlabel('Tscore')
    plt.ylabel('predicted y')

    # Log the correlation plot to WandB
    wandb.log({"correlation_plot": wandb.Image(plt)})
    plt.close()

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
    plt.title('ROC AUC')
    plt.legend(loc="lower right")

    # Log the AUC plot to WandB
    wandb.log({"AUC_plot": wandb.Image(plt)})
    plt.close()  
    
def transfer_LSTM_finetunelastfc(model_dict_path, n_epochs, X_train, y_train, X_test, y_test, input_size, save_model_dir, output_size = 1):
    model_name, ROI_scheme, hidden_size, batch_size, lr, dropout, dropout_layer, SEED, num_heads, num_layers = ParceModelParamsFromPretrainedModel(model_dict_path)

    print(f"model_name: {model_name}, hidden_size: {hidden_size}, batch_size: {batch_size}, lr: {lr}, dropout: {dropout}, dropout_layer: {dropout_layer}, SEED: {SEED}, num_heads: {num_heads}, num_layers: {num_layers}")
    # set initialization
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model_name == 'LSTM':
        model = LSTN(input_size, hidden_size, output_size, dropout=dropout)
    elif model_name == 'BLSTM':
        model = BLSTN(input_size, hidden_size, output_size, dropout=dropout)
    elif model_name == 'LSTMWithAttention':
        model = LSTMWithAttention(input_size, hidden_size, output_size, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'BiLSTMWithAttention':

        model = BiLSTMWithAttention(input_size, hidden_size, output_size, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'LSTMWithTransformer':
        model = LSTMWithTransformer(input_size, hidden_size, output_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'BiLSTMWithTransformer':
        model = BiLSTMWithTransformer(input_size, hidden_size, output_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout, dropout_layer=dropout_layer)    
    else:
        raise ValueError("model_name must be: 'LSTM', 'BLSTM', 'LSTMWithAttention', 'BiLSTMWithAttention', 'LSTMWithTransformer', 'BiLSTMWithTransformer' ")    
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = CustomRSFCMDataset(X_train, y_train, transform=None, target_transform=None)
    test_dataset = CustomRSFCMDataset(X_test, y_test, transform=None, target_transform=None)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(" before freezing: ")
    count_params(model)

    pretrained_dict = torch.load(model_dict_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)
    # Freeze all layers except the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune the fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    print(" after freezing: ")    
    count_params(model)

    test_auc_list = []
    test_accuracy_list = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_auc = 0.0
        train_accuracy = 0.0
        for X, y in tqdm(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            outputs = model(X.float())

            loss = criterion(outputs.squeeze(), y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X.size(0)

            if np.unique(y.detach().cpu().numpy()).shape[0]>1:
                train_auc += roc_auc_score(y.detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())
                
            # train_auc += roc_auc_score(y.detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())
            train_accuracy += accuracy_score(y.detach().cpu().numpy(), torch.round(torch.sigmoid(outputs)).detach().cpu().numpy())
        
        train_loss /= len(train_dataloader)
        train_auc /= len(train_dataloader)
        train_accuracy /= len(train_dataloader)
        
        model.eval()
        test_loss = 0.0
        test_auc = 0.0
        test_accuracy = 0.0
        test_df = pd.read_csv(f'{save_model_dir}/test_df.csv')
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                
                outputs = model(X.float())
                
                loss = criterion(outputs.squeeze(), y.float())
                
                test_loss += loss.item() * X.size(0)
                test_auc += roc_auc_score(y.detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())
                test_accuracy += accuracy_score(y.detach().cpu().numpy(), torch.round(torch.sigmoid(outputs)).detach().cpu().numpy())

        # save for debugging
        test_df['y_pred'] = outputs.detach().cpu().numpy()        
        test_df['y_pred_sigmoid'] = torch.sigmoid(outputs).detach().cpu().numpy()
        test_df['y'] = y.detach().cpu().numpy()
        test_df.to_csv(f'{save_model_dir}/test_df_pred_epoch-{epoch+1}.csv', index=False)
        # save the model
        if (epoch+1) % 5 == 0:
            results_file_name = f'{save_model_dir}/model-{model_name}_{ROI_scheme}_hidden-{hidden_size}_batch-{batch_size}_lr-{lr}_epoch-{epoch+1}_dropout-{dropout}_seed-{SEED}'
            torch.save(model.state_dict(), f'{results_file_name}.pt')

        test_loss /= len(test_dataloader)
        test_auc /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)
        
        print(f"Epoch {epoch+1}/{n_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Accuracy: {test_accuracy:.4f}")
        
        wandb.log({'train_loss': train_loss})
        wandb.log({'train_auc': train_auc})
        wandb.log({'train_accuracy': train_accuracy})

        wandb.log({'test_loss': test_loss})
        wandb.log({'test_auc': test_auc})
        wandb.log({'test_accuracy': test_accuracy})

        test_auc_list.append(test_auc)
        test_accuracy_list.append(test_accuracy)
    print(f"best test auc: {max(test_auc_list):.4f}")
    print(f"average test auc: {np.mean(test_auc_list):.4f}")
    print(f"best test accuracy: {max(test_accuracy_list):.4f}")
    print(f"average test accuracy: {np.mean(test_accuracy_list):.4f}")

    # add the final results to wandb
    log_my_plot(test_df['y'], test_df['y_pred'], test_df['TScore'], test_df['y_pred_sigmoid'])

    return test_auc_list, test_accuracy_list, model

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
    accuracy = correct / total * 100  # Accuracy in percentage
    AUC = roc_auc_score(batch_labels.detach().cpu().numpy(), probas_class_1.detach().cpu().numpy())
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    print(f"ACU score: {AUC:.4f}")
    return avg_loss, accuracy, AUC

def construct_model(model_dict_path):
    model_name, ROI_scheme, hidden_size, batch_size, lr, dropout, dropout_layer, SEED, num_heads, num_layers = ParceModelParamsFromPretrainedModel(model_dict_path)
    print(f"model_name: {model_name}, ROI_scheme: {ROI_scheme}, hidden_size: {hidden_size}, batch_size: {batch_size}, lr: {lr}, dropout: {dropout}, dropout_layer: {dropout_layer}, SEED: {SEED}, num_heads: {num_heads}, num_layers: {num_layers}")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_size = int(ROI_scheme.split('schaefer')[1])
    print(f'input_size: {input_size}')
    output_size = 2

    if model_name == 'LSTM':
        model = LSTN(input_size, hidden_size, output_size, dropout=dropout)
    elif model_name == 'BLSTM':
        model = BLSTN(input_size, hidden_size, output_size, dropout=dropout)
    elif model_name == 'LSTMWithAttention':
        model = LSTMWithAttention(input_size, hidden_size, output_size, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'BiLSTMWithAttention':

        model = BiLSTMWithAttention(input_size, hidden_size, output_size, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'LSTMWithTransformer':
        model = LSTMWithTransformer(input_size, hidden_size, output_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout, dropout_layer=dropout_layer)
    elif model_name == 'BiLSTMWithTransformer':
        model = BiLSTMWithTransformer(input_size, hidden_size, output_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout, dropout_layer=dropout_layer)    
    else:
        raise ValueError("model_name must be: 'LSTM', 'BLSTM', 'LSTMWithAttention', 'BiLSTMWithAttention', 'LSTMWithTransformer', 'BiLSTMWithTransformer' ")   

    model.load_state_dict(torch.load(model_dict_path, map_location=torch.device('cpu')))

    return model

def main(model_dict_path, save_model_dir, y_variable='group'):
    project_dir = '/home/yiyuw/projects/HEAL'
    source_data_set = 'UKB'
    target_data_set = 'HEAL' 

    
    if target_data_set == 'HEAL':
        data_dir = f'{project_dir}/parcellations'
        rs_timepioint = 750
    elif target_data_set == 'UKB':    
        data_dir = f'{project_dir}/ukb_parcellations'
        rs_timepioint = 490
    else: # default to HEAL for now
        data_dir = f'{project_dir}/parcellations'

    # load the pretrained model:
    model = construct_model(model_dict_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_name, ROI_scheme, hidden_size, batch_size, lr, dropout, dropout_layer, SEED, num_heads, num_layers = ParceModelParamsFromPretrainedModel(model_dict_path)

    NROI = int(model_dict_path.split('schaefer')[1].split('_hidden')[0])
    print(f'NROI: {NROI}')
    # parce the model parameters
  
    # set up target data set
    mask_img = f'{project_dir}/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'
    augmented_df = create_labels_df(target_data_set, data_dir, ROI_scheme)
    print(augmented_df)

    # make X
    X, y = [], []
    for i, row in augmented_df.iterrows():
        f = augmented_df.loc[i, 'file']
        X.append(f)
        y.append(augmented_df.loc[i, y_variable])
    y = np.array(y)


    # split based on participants
    subjects_list = np.unique(augmented_df['subject_id'].values).tolist()
    train_sub_idx, test_sub_idx = train_test_split(subjects_list, test_size=0.2, random_state=SEED)
    # find the indices of the subjects in the augmented_df
    test_idx = augmented_df[augmented_df['subject_id'].isin(test_sub_idx)].index
    
    # get the parcel file names per subject
    X_test_sub = np.array(X)[test_idx.astype(int)]

    # # make X_train and X_test
    network_list = ['Vis','SomMot','DorsAttn','SalVent','Limbic','Cont','Default', None]
    loss_list, AUC_list, accuracy_list = [], [], []

    for network in network_list:
        print(f'ablating: network: {network}')
        X_test = ParcelFileToTensor(X_test_sub, NROI, rs_timepoint=rs_timepioint, network=network)
        y_test = torch.tensor(y[test_idx.astype(int)])
        
        # save the train_df, and test_df
        test_df = augmented_df.loc[test_idx].to_csv(f'{save_model_dir}/test_df.csv', index=False)
        
        test_dataset = CustomRSFCMDataset(X_test, y_test, transform=None, target_transform=None)
        
        # Step 3: Create a DataLoader 
        test_dataloader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)


        avg_loss, accuracy, AUC = evaluate_pretrained_model(model, test_dataloader, device)
        results_file_name = f'{save_model_dir}/model-{model_name}_{ROI_scheme}_hidden-{hidden_size}_batch-{batch_size}_lr-{lr}_dropout-{dropout}_seed-{SEED}_network-{network}'
        loss_list.append(avg_loss)
        accuracy_list.append(accuracy)
        AUC_list.append(AUC)
    network_results = pd.DataFrame({'network': network_list, 'test_loss': loss_list, 'test_accuracy': accuracy_list, 'test_AUC': AUC_list})
    network_results.to_csv(f'{save_model_dir}/ablating_network_results_SEED-{SEED}.csv', index=False)



if __name__ == "__main__":
    for seed in [30, 35, 42, 50, 60, 70, 80, 88, 90, 100]:
        model_dict_name = f'model-LSTMWithTransformer_ROI-schaefer200_hidden-256_batch-16_lr-0.0001_epoch-100_dropout-0.5_dropoutlayer-0.2_heads-8_layers-2_seed-{seed}.pt'
        model_dir = '/home/yiyuw/projects/HEAL/saved_models/heal_rs_lstm_classification_v3/*/'
        print(glob.glob(model_dir + model_dict_name))
        model_dict_path = glob.glob(model_dir + model_dict_name)[0]
        save_dir = '/home/yiyuw/projects/HEAL/saved_models/heal_rs_lstm_classification_v3/'
        print(f'running: {model_dict_path}')
        main(model_dict_path, save_dir,'group')


    