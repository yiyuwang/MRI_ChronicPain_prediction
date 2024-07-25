#!/usr/bin/env python

'''
run LSTM models on resting state
'''

#import pandas as pd
import glob, sys
import numpy as np
import pandas as pd
import os

from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
from nilearn.maskers import NiftiMasker,NiftiLabelsMasker
from nilearn import plotting
from sklearn import metrics
import warnings
import copy
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn as nn

import re
import math
import time
import glob
import random
import string
import collections

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupShuffleSplit

from tqdm import tqdm

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
torch.cuda.is_available()


SEED = 2024
# local directory
project_dir = '/home/yiyuw/projects/HEAL'

# # directory on sherlock
# project_dir = '/scratch/users/yiyuw/Projects/HEAL_prediction'


data_dir = f'{project_dir}/parcellations'
confounds_dir = f'{project_dir}/confounds'
save_model_dir = f'{project_dir}/saved_models'
mask_img = f'{project_dir}/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'


ROI_scheme = sys.argv[1]
print(f'using ROI scheme - {ROI_scheme}')


# calculate reliability between two runs:

warnings.filterwarnings('ignore')


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
    
def ParcelFileToTensor(parcel_file_list, transform = None, network = None):
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
        
        if par_df.shape[0] != 750:
            print(f'Error: {parcellation_path} has {par_df.shape[0]} time points, not 750. Skipping...')
            continue
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.2, dropout_layer=0.2, dropout_attention=0.2):
        super(LSTMWithTransformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )

        if dropout_layer != 0 :
            self.dropout = nn.Dropout(dropout_layer)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pass through the transformer encoder
        transformer_out = self.transformer(lstm_out)
        
        # Apply global average pooling
        context_vector = torch.mean(transformer_out, dim=1)

        # apply dropout
        if hasattr(self, 'dropout') and dropout_attention != 0:
            context_vector = self.dropout(context_vector)
        
        # Output layer
        output = self.fc(context_vector)
        return output
    
class BiLSTMWithTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.2, dropout_layer=0.2, dropout_attention=0.2):
        super(BiLSTMWithTransformer, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim*2, nhead=num_heads),
            num_layers=num_layers
        )
        if dropout_layer != 0 :
            self.dropout = nn.Dropout(dropout_layer)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        bilstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Pass through the transformer encoder
        transformer_out = self.transformer(bilstm_out)
        
        # Apply global average pooling
        context_vector = torch.mean(transformer_out, dim=1)
        
        # apply dropout
        if hasattr(self, 'dropout') and dropout_attention != 0:
            context_vector = self.dropout(context_vector)
        # Output layer
        output = self.fc(context_vector)
        return output
    
        
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(context_vector)
        return output
    
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(BiLSTMWithAttention, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, droupout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        bilstm_out, (h_n, c_n) = self.bilstm(x)
        
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(bilstm_out), dim=1)
        context_vector = torch.sum(attention_weights * bilstm_out, dim=1)
        
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
    
    


def train_LSTN(model_name, X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, n_epochs, batch_size, lr, y_variable, dropout=0.2, dropout_layer=0.2, dropout_attention=0.2, num_heads=4, num_layers=2):
    if model_name == 'LSTM':
        model = LSTN(input_size, hidden_size, output_size)
    elif model_name == 'BLSTM':
        model = BLSTN(input_size, hidden_size, output_size)
    elif model_name == 'LSTMWithAttention':
        model = LSTMWithAttention(input_size, hidden_size, output_size)
    elif model_name == 'BiLSTMWithAttention':

        model = BiLSTMWithAttention(input_size, hidden_size, output_size)
    elif model_name == 'LSTMWithTransformer':
        model = LSTMWithTransformer(input_size, hidden_size, output_size, num_heads=num_heads, num_layers=num_layers)
    elif model_name == 'BiLSTMWithTransformer':
        model = BiLSTMWithTransformer(input_size, hidden_size, output_size, num_heads=num_heads, num_layers=num_layers)    
    else:
        raise ValueError("model_name must be: 'LSTM', 'BLSTM', 'LSTMWithAttention', 'BiLSTMWithAttention'")    
      
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = CustomRSFCMDataset(X_train, y_train, transform=None, target_transform=None)
    test_dataset = CustomRSFCMDataset(X_test, y_test, transform=None, target_transform=None)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
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
        
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                
                outputs = model(X.float())
                loss = criterion(outputs.squeeze(), y.float())
                
                test_loss += loss.item() * X.size(0)
                test_auc += roc_auc_score(y.detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())
                test_accuracy += accuracy_score(y.detach().cpu().numpy(), torch.round(torch.sigmoid(outputs)).detach().cpu().numpy())
        
        test_loss /= len(test_dataloader)
        test_auc /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)
        
        print(f"Epoch {epoch+1}/{n_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Accuracy: {test_accuracy:.4f}")

        test_auc_list.append(test_auc)
        test_accuracy_list.append(test_accuracy)
    print(f"best test auc: {max(test_auc_list):.4f}")
    print(f"average test auc: {np.mean(test_auc_list):.4f}")
    print(f"best test accuracy: {max(test_accuracy_list):.4f}")
    print(f"average test accuracy: {np.mean(test_accuracy_list):.4f}") 

    
        
    return test_auc_list, test_accuracy_list, model

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
        print(s, run)
        df = CP_df[CP_df['subject_id']==s]
        df['run'] = run
        df['file'] = f
        augmented_df = pd.concat([augmented_df, df])
augmented_df = augmented_df.reset_index(drop=True)

cv_test_auc_list, cv_test_accuracy_list = [], []

n_split = 5
train_split = 0.75
augmented_df.reset_index(drop=True, inplace=True)

# make X
X, y = [], []
for i, row in augmented_df.iterrows():
    f = augmented_df.loc[i, 'file']
    X.append(f)
    y.append(augmented_df.loc[i, 'group'])

y = np.array(y)

subjects_list = augmented_df['subject_id'].values.tolist()
gss = GroupShuffleSplit(n_splits=n_split, train_size=train_split, random_state=SEED)
hidden_size = int(sys.argv[2])
batch_size = int(sys.argv[3])
model_name = sys.argv[4]
SEED = int(sys.argv[5])
if 'schaefer' in ROI_scheme:
    NROI = int(ROI_scheme.split('schaefer')[1])
elif 'msdl' in ROI_scheme:
    NROI = 39
else:
    raise ValueError('ROI scheme not recognized')    
for i, (train_idx, test_idx) in enumerate(gss.split(X, y, subjects_list)):
    print(f'---------------------- fold {i+1} ---------------------------')

    X_train_sub = np.array(X)[train_idx.astype(int)]
    X_test_sub = np.array(X)[test_idx.astype(int)]
    X_train = ParcelFileToTensor(X_train_sub)
    X_test = ParcelFileToTensor(X_test_sub)

    y_train = torch.tensor(y[train_idx.astype(int)])
    y_test = torch.tensor(y[test_idx.astype(int)])

    test_auc_list, test_accuracy_list, model = train_LSTN(model_name, X_train, y_train, X_test, y_test, input_size=NROI, hidden_size=hidden_size, output_size=1, n_epochs=50, batch_size=batch_size, lr=0.001, y_variable='group')
    cv_test_auc_list.append(test_auc_list)
    cv_test_accuracy_list.append(test_accuracy_list)

    results_file_name = f'{save_model_dir}/cv-{i+1}_model-{model_name}_{ROI_scheme}_hidden-{hidden_size}_batch-{batch_size}_lr0.001_fold{n_split}_seed{SEED}'
    torch.save(model.state_dict(), f'{results_file_name}.pt')
    cv_test_df = pd.DataFrame({'auc':cv_test_auc_list[i], 'accuracy':cv_test_accuracy_list[i], 'epoch':range(len(cv_test_auc_list[i]))}).to_csv(f'{results_file_name}.csv')

print(f"total auc: {np.mean(cv_test_auc_list):.4f}")
print(f"total accuracy: {np.mean(cv_test_accuracy_list):.4f}")

# python HEAL_rs_lstn.py 'schaefer100' 128 32 'LSTMWithTransformer' 48

