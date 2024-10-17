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
    
def ParcelFileToTensor(parcel_file_list, NROI, rs_timepoint=750, transform = None, network = None):
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

def train_LSTM(model_name, train_dataloader, test_dataloader, input_size, hidden_size, output_size, n_epochs, batch_size, lr, y_variable, save_model_dir, dropout=0.2, dropout_layer=0.2, num_heads=4, num_layers=2, SEED=42):
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
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    count_params(model)
    
    test_auc_list = []
    test_accuracy_list = []
    y_preds_list, y_preds_prob_list, y_train_list = [], [], []
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
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X.size(0)

            # Probability for class 1 (positive class)
            probas_class_1 = F.softmax(outputs, dim=1)[:, 1]  # Extract probability for class 1
            y_preds_list.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            y_preds_prob_list.append(probas_class_1.detach().cpu().numpy())
            y_train_list.append(y.detach().cpu().numpy())

        y_preds_list_flattened = [item for sublist in y_preds_list for item in sublist]
        y_preds_prob_list_flattened = [item for sublist in y_preds_prob_list for item in sublist]
        y_train_list_flattened = [item for sublist in y_train_list for item in sublist]

        train_auc = roc_auc_score(y_train_list_flattened, y_preds_prob_list_flattened)
        train_accuracy = accuracy_score(y_train_list_flattened, y_preds_list_flattened)
        train_loss /= len(train_dataloader)
        
        
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
                
                loss = criterion(outputs, y)
                
                test_loss += loss.item() * X.size(0)
                probas_class_1 = F.softmax(outputs, dim=1)[:, 1]  # Extract probability for class 1
    
                test_auc += roc_auc_score(y.detach().cpu().numpy(), probas_class_1.detach().cpu().numpy())
                test_accuracy += accuracy_score(y.detach().cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())

        # save for debugging
        test_df['y_pred'] = torch.argmax(outputs, dim=1).cpu().numpy()      
        test_df['y_pred_prob'] = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        test_df['y'] = y.detach().cpu().numpy()
        test_df.to_csv(f'{save_model_dir}/test_df_pred_epoch-{epoch+1}.csv', index=False)
        # save the model
        if (epoch+1) % 20 == 0:
            results_file_name = f'{save_model_dir}/model-{model_name}_ROI-{ROI_scheme}_hidden-{hidden_size}_batch-{batch_size}_lr-{lr}_epoch-{epoch+1}_dropout-{dropout}_dropoutlayer-{dropout_layer}_heads-{num_heads}_layers-{num_layers}_seed-{SEED}'
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

    log_my_plot(y_test=test_df['y'], y_pred=test_df['y_pred'], y_pred_sigmoid=test_df['y_pred_prob'])
 
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
                if pd.read_csv(file_path).shape[0] != 490:
                    print(f'Error: {file_path} has {pd.read_csv(file_path).shape[0]} time points, not 490. Skipping...')
                    augmented_df = augmented_df[augmented_df['eid']!=s]    
                    continue
            augmented_df.reset_index(drop=True, inplace=True)
            # rename eid column to subject_id
            augmented_df.rename(columns={'eid':'subject_id'}, inplace=True)
            # save to csv
            augmented_df.to_csv(f'{project_dir}/ukb_labels/ukb_labels_filecheck.csv', index=False)
        else:
            augmented_df = pd.read_csv(f'{project_dir}/ukb_labels/ukb_labels_filecheck.csv')
    return augmented_df

def main(
    output_size = 2,
    hidden_size = 128,
    batch_size = 16,
    model_name = 'LSTN',
    ROI_scheme = 'schaefer100',
    n_ep = 10,
    lr=0.001, 
    y_variable='group', 
    dropout=0.2, 
    dropout_layer=0.2, 
    num_heads=4, 
    num_layers=2,
    SEED = 2024):


    
    project_dir = '/home/yiyuw/projects/HEAL'
    source_data_set = 'UKB'
    target_data_set = 'HEAL'

    if source_data_set == 'HEAL':
        data_dir = f'{project_dir}/parcellations'
        rs_timepioint = 750
    elif source_data_set == 'UKB':    
        data_dir = f'{project_dir}/ukb_parcellations'
        rs_timepioint = 490
    else:
        data_dir = f'{project_dir}/parcellations'

    mask_img = f'{project_dir}/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'
    augmented_df = create_labels_df(source_data_set,data_dir, ROI_scheme)
    cv_test_auc_list, cv_test_accuracy_list = [], []
    print(augmented_df)
    # make X
    X, y = [], []
    for i, row in augmented_df.iterrows():
        f = augmented_df.loc[i, 'file']
        X.append(f)
        y.append(augmented_df.loc[i, y_variable])
    y = np.array(y)

    
    if 'schaefer' in ROI_scheme:
        NROI = int(ROI_scheme.split('schaefer')[1])
    elif 'msdl' in ROI_scheme:
        NROI = 39
    else:
        raise ValueError('ROI scheme not recognized') 
    
    # Initialize a new wandb run
    project_name = f"{source_data_set}_rs_lstm_pretrain_v3"
    config_args = {'hidden_size': hidden_size, 
    'batch_size': batch_size, 
    'model_name': model_name, 
    'ROI_scheme': ROI_scheme, 
    'y_variable': y_variable,
    'lr': lr, 
    'num_heads': num_heads, 
    'num_layers': num_layers, 
    'dropout': dropout, 
    'dropout_layer': dropout_layer, 
    'SEED': SEED, 
    'n_ep': n_ep}
    run = wandb.init(config=config_args,
        project=project_name,
        entity='yiyuwang1')
    print(f'run_name: {run.name}')
    save_model_dir = get_model_dir(f"{project_name}/" + run.name, project_dir)
    print(f'save_model_dir: {save_model_dir}')

    # split based on participants
    subjects_list = np.unique(augmented_df['subject_id'].values).tolist()
    train_sub_idx, test_sub_idx = train_test_split(subjects_list, test_size=0.2, random_state=SEED)
    # find the indices of the subjects in the augmented_df
    train_idx = augmented_df[augmented_df['subject_id'].isin(train_sub_idx)].index
    test_idx = augmented_df[augmented_df['subject_id'].isin(test_sub_idx)].index
    
    # get the parcel file names per subject
    X_train_sub = np.array(X)[train_idx.astype(int)]
    X_test_sub = np.array(X)[test_idx.astype(int)]

    # # make X_train and X_test
    X_train = ParcelFileToTensor(X_train_sub, NROI, rs_timepoint=rs_timepioint)
    X_test = ParcelFileToTensor(X_test_sub, NROI, rs_timepoint=rs_timepioint)

    y_train = torch.tensor(y[train_idx.astype(int)])
    y_test = torch.tensor(y[test_idx.astype(int)])
    
    
    # save the train_df, and test_df
    train_df = augmented_df.loc[train_idx].to_csv(f'{save_model_dir}/train_df.csv', index=False)
    test_df = augmented_df.loc[test_idx].to_csv(f'{save_model_dir}/test_df.csv', index=False)


    train_dataset = CustomRSFCMDataset(X_train, y_train, transform=None, target_transform=None)
    test_dataset = CustomRSFCMDataset(X_test, y_test, transform=None, target_transform=None)
    
    # Step 3: Create a DataLoader 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)

    test_auc_list, test_accuracy_list, model = train_LSTM(model_name, train_dataloader, test_dataloader, input_size=NROI, hidden_size=hidden_size, output_size=output_size, n_epochs=n_ep, batch_size=batch_size, lr=lr, y_variable=y_variable, save_model_dir=save_model_dir, dropout=dropout, dropout_layer=dropout_layer, num_heads=num_heads, num_layers=num_layers, SEED=SEED) 

    results_file_name = f'{save_model_dir}/model-{model_name}_ROI-{ROI_scheme}_hidden-{hidden_size}_batch-{batch_size}_lr-{lr}_epoch-{n_ep}_dropout-{dropout}_dropoutlayer-{dropout_layer}_heads-{num_heads}_layers-{num_layers}_seed-{SEED}'
    # torch.save(model.state_dict(), f'{results_file_name}.pt')
    cv_test_df = pd.DataFrame({'auc':test_auc_list, 'accuracy':test_accuracy_list, 'epoch':range(len(test_auc_list))}).to_csv(f'{results_file_name}.csv')


    print(f"total auc: {np.mean(test_auc_list):.4f}")
    print(f"total accuracy: {np.mean(test_accuracy_list):.4f}")

    # python HEAL_rs_lstn.py 'schaefer100' 128 32 'LSTMWithTransformer' 48
    wandb.finish()

if __name__ == "__main__":
   #main(label='group',base_model_name='swin', num_epochs = 50)
   # create a dictionary of parameters
    # params_dict = {
    # 'hidden_size': [128, 256],
    # 'batch_size': [8, 16],
    # 'model_name': ['BiLSTMWithTransformer', 'LSTMWithTransformer'],
    # 'ROI_scheme': ['schaefer400', 'schaefer1000', 'schaefer200'],
    # 'lr': [0.001, 0.0001],
    # 'num_heads': [4],
    # 'num_layers': [2],
    # 'dropout': [0.1, 0.2, 0.3],
    # 'dropout_layer': [0.2],
    # 'SEED': [42, 5],
    # }
    params_dict = {
    'hidden_size': [256],
    'batch_size': [16],
    'model_name': ['LSTMWithTransformer'],
    'ROI_scheme': ['schaefer200'],
    'lr': [0.0001],
    'num_heads': [8],
    'num_layers': [2],
    'dropout': [0.5],
    'dropout_layer': [0.2],
    'SEED': [50, 60, 70, 80, 88, 90, 100],
    # 'SEED': [30, 35, 42],
    'y_variable': ['sex']
    }

    output_size = 2
    print("this is pretraining a classifier using UKB resting state data")
    print(f"Make sure output size is 2: Output_size = {output_size}")
    from itertools import product
    total_iteration = len(list(product(*params_dict.values())))
    print(f'Number of iterations: {total_iteration}')

    for i, params in enumerate(product(*params_dict.values())):
        print(f'---------Iteration {i+1} / {total_iteration} --------')
        hidden_size, batch_size, model_name, ROI_scheme, lr, num_heads, num_layers, dropout, dropout_layer, SEED, y_variable = params
        print(f'y_variable: {y_variable}, hidden_size: {hidden_size}, batch_size: {batch_size}, model_name: {model_name}, ROI_scheme: {ROI_scheme}, lr: {lr}, num_heads: {num_heads}, num_layers: {num_layers}, dropout: {dropout}, dropout_layer: {dropout_layer}, SEED: {SEED}')
        
        main(
        output_size = 2,
        hidden_size = hidden_size,
        batch_size = batch_size,
        model_name = model_name,
        ROI_scheme = ROI_scheme,
        n_ep = 100,
        lr=lr, 
        y_variable=y_variable, 
        dropout=dropout, 
        dropout_layer=dropout_layer, 
        num_heads=num_heads, 
        num_layers=num_layers,
        SEED = SEED)


    