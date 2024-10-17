#!/usr/bin/env python

'''
run LSTM models on resting state of HEAL from scratch
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
from torch.utils.data import Dataset, DataLoader, BatchSampler
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
    
def ParcelFileToTensor(parcel_file_list, NROI, transform = None, network = None, drop_network = False):
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
            raise ValueError(f'Error: {parcellation_path} has {par_df.shape[0]} time points, not 750')
        else: 
            # Lesion first:
            if network:
                x_cols = [col for col in par_df.columns if 'Networks' in col]
                zero_cols = [col for col in x_cols if (network in col)]
                if drop_network:
                    print(f'Dropping {network} network')
                    par_df.drop(zero_cols, axis=1, inplace=True)
                else:
                    print(f"Lesioning {network} network")
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Positional Encoding for timepoints
# ref: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformer Encoder Layer with Dropout
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def generate_square_subsequent_mask(sz):
    """Generates a mask to prevent attending to future tokens (for autoregressive models)."""
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1).float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class MRIRestingStatePositionalEmbeddingTransformer(nn.Module):
    def __init__(self, 
                 NROI,  # Number of ROIs (input feature dimensions)
                 d_model=512,  # Embedding dimension
                 nhead=8,  # Number of attention heads
                 num_layers=6,  # Number of transformer layers
                 dim_feedforward=2048,  # Dimension of feed-forward layers
                 dropout=0.1,  # Dropout rate
                 max_len=5000,  # Maximum sequence length
                 num_classes=10,  # Number of classes for classification
                 pooling='mean'  # Pooling strategy: 'mean' or 'max'
                 ):
        super(MRIRestingStatePositionalEmbeddingTransformer, self).__init__()
        
        # Linear projection from NROI to d_model (embedding dimension)
        self.input_projection = nn.Linear(NROI, d_model)
        
        # Positional embedding layer
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # Pre-normalization (LayerNorm before Transformer)
        self.pre_norm = nn.LayerNorm(d_model)  # Normalizing the input embeddings

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Post-normalization (LayerNorm after Transformer)
        self.post_norm = nn.LayerNorm(d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Pooling layer for sequence summarization
        if pooling == 'mean':
            self.pooling_layer = nn.AdaptiveAvgPool1d(1)  # Reduce to 1 timepoint
        elif pooling == 'max':
            self.pooling_layer = nn.AdaptiveMaxPool1d(1)
        
        # Final classification layer
        self.fc_classification = nn.Linear(d_model, num_classes)
        
    def forward(self, x, src_mask=None):
        # x has shape (batch_size, timepoints, NROI) -> convert to (batch_size, timepoints, d_model)
        x = self.input_projection(x)
        
        # Add positional embeddings
        x = self.positional_encoding(x)
        
        # Apply pre-normalization
        x = self.pre_norm(x)  # Normalize before passing to Transformer
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_mask)
        
        # Apply post-normalization
        x = self.post_norm(x)  # Normalize after Transformer
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transpose for pooling: (batch_size, timepoints, d_model) -> (batch_size, d_model, timepoints)
        x = x.permute(0, 2, 1)
        
        # Apply pooling: (batch_size, d_model, timepoints) -> (batch_size, d_model, 1)
        x = self.pooling_layer(x)
        
        # Remove the singleton timepoint dimension: (batch_size, d_model, 1) -> (batch_size, d_model)
        x = x.squeeze(-1)
        
        # Final classification layer
        output = self.fc_classification(x)
        
        return output


def my_binarize(y, threshold=65):
    y = np.array(y)
    y_bin = np.zeros(len(y))
    y_bin[y >= threshold] = 1
    return y_bin

def log_my_plot(y_test, y_pred, Tscore, y_pred_sigmoid):
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
    

def train_model(model_name, train_dataloader, test_dataloader, input_size, output_size, n_epochs, batch_size, lr, y_variable, save_model_dir, d_model=512, dim_feedforward = 2048, dropout=0.2, num_heads=4, num_layers=2, SEED=42):
    # set initialization
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Instantiate model
    if model_name == 'MRIRestingStatePositionalEmbeddingTransformer':
        model = MRIRestingStatePositionalEmbeddingTransformer(NROI=input_size, d_model=d_model, nhead=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, max_len=750, num_classes=output_size, pooling='mean')
    else:
        raise ValueError('model not created yet')
    
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

            # test mask:
            # mask is probably not useful given that all the input are same length...?
            # src_mask = generate_square_subsequent_mask(X.size(1)).to(device)
            # print('this training use src autoregression mask')
            # outputs = model(X.float(), src_mask=src_mask)

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
            results_file_name = f'{save_model_dir}/model-{model_name}_ROI-{ROI_scheme}_batch-{batch_size}_lr-{lr}_epoch-{epoch+1}_dropout-{dropout}_dimfeedforward-{dim_feedforward}_d_model-{d_model}_heads-{num_heads}_layers-{num_layers}_seed-{SEED}'
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

    log_my_plot(y_test=test_df['y'],Tscore=test_df['TScore'], y_pred=test_df['y_pred'], y_pred_sigmoid=test_df['y_pred_prob'])

    # get the last batch in train_dataloader as sanity check:
    X, y = next(iter(train_dataloader))
    X = X.to(device)
    y = y.to(device)
    outputs = model(X.float())
    probas_class_1 = F.softmax(outputs, dim=1)[:, 1]  # Extract probability for class 1
    print('sanity check train auc: ',roc_auc_score(y.detach().cpu().numpy(), probas_class_1.detach().cpu().numpy()))
    print('sanity check train accuracy: ', accuracy_score(y.detach().cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy()))
 
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


def main(
    model_name = 'SpaceTimeFormer',
    output_size = 2,
    batch_size = 16,
    ROI_scheme = 'schaefer100',
    n_ep = 100,
    lr=0.001, 
    y_variable='group', 
    dropout=0.2, 
    d_model=512,
    dim_feedforward = 2048, 
    num_heads=4, 
    num_layers=2,
    SEED = 2024):

    
    project_dir = '/home/yiyuw/projects/HEAL'

    data_dir = f'{project_dir}/parcellations'
    confounds_dir = f'{project_dir}/confounds'
    mask_img = f'{project_dir}/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'


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
    augmented_df.to_csv(f'{project_dir}/HEAL_augmented_df.csv', index=False)
    cv_test_auc_list, cv_test_accuracy_list = [], []

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
    project_name = "heal_rs_transformer_classification_v1"
    config_args = {'y_variable': y_variable, 'batch_size': batch_size, 'model_name': model_name, 'ROI_scheme': ROI_scheme, 'lr': lr, 'd_model': d_model, 'num_heads': num_heads, 'num_layers': num_layers, 'dropout': dropout, 'dim_feedforward': dim_feedforward, 'SEED': SEED, 'n_ep': n_ep}
    run = wandb.init(config=config_args,
        project=project_name,
        entity='yiyuwang1')
    print(f'run_name: {run.name}')
    save_model_dir = get_model_dir(f"{project_name}/" + run.name, project_dir)
    print(f'save_model_dir: {save_model_dir}')

    # split based on participants
    subjects_list = np.unique(augmented_df['subject_id'].values).tolist()
    train_sub_idx, test_sub_idx = train_test_split(subjects_list, test_size=0.2, random_state=SEED)
    
    # Creete datasets and dataloaders
    # find the indices of the subjects in the augmented_df
    train_idx = augmented_df[augmented_df['subject_id'].isin(train_sub_idx)].index
    test_idx = augmented_df[augmented_df['subject_id'].isin(test_sub_idx)].index
    
    # get the parcel file names per subject
    X_train_sub = np.array(X)[train_idx.astype(int)]
    X_test_sub = np.array(X)[test_idx.astype(int)]

    # make X_train and X_test
    X_train = ParcelFileToTensor(X_train_sub, NROI)
    X_test = ParcelFileToTensor(X_test_sub, NROI)
    input_size = X_train.shape[2]
    print(f'input_size: {input_size}')
    print(X_train.shape, X_test.shape)

    y_train = torch.tensor(y[train_idx.astype(int)])
    y_test = torch.tensor(y[test_idx.astype(int)])

    train_dataset = CustomRSFCMDataset(X_train, y_train, transform=None, target_transform=None)
    test_dataset = CustomRSFCMDataset(X_test, y_test, transform=None, target_transform=None)
    
    # Step 3: Create a DataLoader 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)

    # save the train_df, and test_df
    train_df = augmented_df.loc[train_idx].to_csv(f'{save_model_dir}/train_df.csv', index=False)
    test_df = augmented_df.loc[test_idx].to_csv(f'{save_model_dir}/test_df.csv', index=False)

    test_auc_list, test_accuracy_list, model = train_model(model_name, train_dataloader, test_dataloader, input_size=input_size, output_size=output_size, n_epochs=n_ep, batch_size=batch_size, lr=lr, y_variable=y_variable, save_model_dir=save_model_dir, dropout=dropout, dim_feedforward = dim_feedforward, d_model=d_model, num_heads=num_heads, num_layers=num_layers, SEED=SEED) 

    results_file_name = f'{save_model_dir}/model-{model_name}_ROI-{ROI_scheme}_batch-{batch_size}_lr-{lr}_epoch-{n_ep}_dropout-{dropout}_dimfeedforward-{dim_feedforward}_d_model-{d_model}_heads-{num_heads}_layers-{num_layers}_seed-{SEED}'
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
    'batch_size': [16],
    'model_name': ['MRIRestingStatePositionalEmbeddingTransformer'],
    'ROI_scheme': ['schaefer400'],
    'lr': [0.0001],
    'd_model': [512],
    'num_heads': [8],
    'num_layers': [6],
    'dropout': [0.5],
    'dim_feedforward': [1024],
    'SEED': [42, 105, 104],
    'y_variable': ['group']
    }
    print("this is a classifcation task. Check the output size is the number of classes...")
    output_size = 2
    from itertools import product
    total_iteration = len(list(product(*params_dict.values())))
    print(f'Number of iterations: {total_iteration}')

    for i, params in enumerate(product(*params_dict.values())):
    
        print(f'---------Iteration {i+1} / {total_iteration} --------')
        batch_size, model_name, ROI_scheme, lr, d_model, num_heads, num_layers, dropout, dim_feedforward, SEED, y_variable = params
        print(f'y_variable: {y_variable}, output_size: {output_size}, batch_size: {batch_size}, model_name: {model_name}, ROI_scheme: {ROI_scheme}, lr: {lr}, d_model: {d_model}, num_heads: {num_heads}, num_layers: {num_layers}, dropout: {dropout}, dim_feedforward: {dim_feedforward}, SEED: {SEED}')
        

        main(
        output_size = 2,
        batch_size = batch_size,
        model_name = model_name,
        ROI_scheme = ROI_scheme,
        n_ep = 200,
        lr=lr, 
        y_variable=y_variable, 
        dropout=dropout, 
        dim_feedforward = dim_feedforward, 
        d_model=d_model,
        num_heads=num_heads, 
        num_layers=num_layers,
        SEED = SEED)


    