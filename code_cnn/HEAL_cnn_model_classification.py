#!/usr/bin/env python

'''
run cnn on 3d mri data: pressure beta, or t1
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import nibabel as nib
import numpy as np
from sklearn.metrics import r2_score
import re, glob, os

# Define CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, output_dim, input_shape):
        super(CNN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        flattened_size = self._get_flattened_size(input_shape)
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _get_flattened_size(self, sample_shape):
        x = torch.zeros(1, *sample_shape)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        return x.numel()


# Define the FMRI_Dataset class
class FMRI_Dataset(Dataset):
    def __init__(self, dataframe, label_column):
        self.dataframe = dataframe
        self.label_column = label_column
        
        # Preload all images and labels into memory
        self.images = []
        self.labels = []
        
        for idx in range(len(dataframe)):
            file_path = dataframe.iloc[idx]['nifti_path']
            img = nib.load(file_path).get_fdata()
            img = np.expand_dims(img, axis=0)  # Add channel dimension
            img_tensor = torch.tensor(img, dtype=torch.float32)
            self.images.append(img_tensor)
            
            label = dataframe.iloc[idx][label_column]
            label_tensor = torch.tensor(label, dtype=torch.float32)  # Use float tensor for regression
            self.labels.append(label_tensor)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def create_dataloaders(dataframe, label_column, batch_size=16, shuffle=True, val_split=0.2, test_split=0.1):
    print('batch_size=', batch_size, ',val_split=', val_split, ',test_split=', test_split)
    dataset = FMRI_Dataset(dataframe, label_column)
    
    # Calculate lengths for train, val, and test splits
    test_len = int(test_split * len(dataset))
    val_len = int(val_split * len(dataset))
    train_len = len(dataset) - val_len - test_len
    
    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    
    return train_loader, val_loader, test_loader


# Define the Linear Regression Network
class LinearRegressionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        running_corr = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs).squeeze()  # Squeeze to remove extra dimensions
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)

            running_corr += np.corrcoef(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())[0, 1]
        
        epoch_corr = running_corr / total_samples
        epoch_loss = running_loss / total_samples
        
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs).squeeze()  # Squeeze to remove extra dimensions
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_outputs.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_r2 = r2_score(val_labels, val_outputs)
        val_corr = np.corrcoef(val_labels, val_outputs)[0, 1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, train_corr: {epoch_corr:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}, Val Corr: {val_corr:.4f}')
    model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_r2 = 0.0
    test_corr = 0.0
    test_outputs = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()  # Squeeze to remove extra dimensions
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            test_r2 += r2_score(labels.cpu().numpy(), outputs.cpu().numpy())
            test_corr += np.corrcoef(labels.cpu().numpy(), outputs.cpu().numpy())[0, 1]
            test_outputs.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_r2 /= len(test_loader)
    test_corr /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}, Test Corr: {test_corr:.4f}')
    return test_loss, test_r2, test_corr, test_outputs, test_labels

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


def fine_tune_cnn3d(pretrained_model, new_input_dim, new_output_dim, latent_dim=512):
    # Load the model state
    model = pretrained_model
    
    # Freeze all layers except the last two fully connected layers
    for param in model.parameters():
        param.requires_grad = False
    

    model.fc1 = nn.Linear(new_input_dim, latent_dim)  # Assuming you still want 512 neurons in the second-to-last layer
    model.fc2 = nn.Linear(latent_dim, new_output_dim)  # New output dimension
   
    # Unfreeze the last two fully connected layers
    model.fc1.requires_grad = True
    model.fc2.requires_grad = True

    return model

def get_path_and_shape(which_data, project_dir):
    if which_data == 'pressure':
        data_shape = (99, 117, 95)
        data_dir = f'{project_dir}/model1_stimulation/1stlvl/*/'
        data_file_name = 'sub-*_stat-z_reg-stimulation_gm_masked.nii.gz'
    elif which_data == 't1_1mm':
        data_shape = (193, 229, 193)
        data_dir = f'{project_dir}/MRI_structural_1mm/*/ses-01/anat/'
        data_file_name = 'sub-*_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
    elif which_data == 't1_2mm':
        data_shape = (91, 109, 91)
        data_dir = f'{project_dir}/MRI_structural_2mm/*/ses-01/anat/'
        data_file_name = 'sub-*_ses-01_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz'
    elif which_data == 'ukb_t1':
        data_shape = (182, 218, 182)
        data_dir = f'{project_dir}/ukb_t1/*/ses-01/anat/'
        data_file_name = 'sub-*_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
    else:
        raise Exception('which_data must be either "pressure" or "t1_1mm" or "t1_2mm", "ukb_t1" ')
    
    return data_dir, data_file_name, data_shape

def get_nifti_path(which_data, CP_df, project_dir):
    data_dir, data_file_name, data_shape = get_path_and_shape(which_data, project_dir)
    print('data_dir:', data_dir, '\ndata_file_name:', data_file_name, '\ndata_shape:', data_shape)
    # include rest run 1 and run 2
    augmented_df = pd.DataFrame(columns=CP_df.columns.to_list() + ['nifti_path'])

    
    for f in np.sort(glob.glob(os.path.join(data_dir, data_file_name))):
        if nib.load(f).shape != data_shape:
            print(f'Error: {f} has shape {nib.load(f).shape} . Skipping...')
            continue
        else:
            s = os.path.basename(f).split('sub-')[1].split('_')[0]
            df = CP_df[CP_df['subject_id']==s]
            df['nifti_path'] = f
            augmented_df = pd.concat([augmented_df, df])

    augmented_df.reset_index(drop=True, inplace=True)
    return augmented_df 

def get_ukb_nifti_path(project_dir, create=True):
    if create:
        print('making new labels file for ukb')
        augmented_df = pd.read_csv(f'{project_dir}/ukb_labels/ukb_labels.csv')
        augmented_df['nifti_path'] = None
        for s in augmented_df['eid']:
            file_path = f'{project_dir}/uk_biobank/sub-{s}/ses-2_0/anat/sub-{s}_ses-2_0_desc-preproc_T1w_brain_space-1mm_MNI.nii.gz'
            augmented_df.loc[augmented_df['eid']==s, 'nifti_path'] = file_path
            # if there is no nifti file, remove the row
            if not os.path.exists(file_path):
                augmented_df = augmented_df[augmented_df['eid']!=s]
        augmented_df.reset_index(drop=True, inplace=True)
        # save to csv
        augmented_df.to_csv(f'{project_dir}/ukb_labels/ukb_labels_filecheck.csv', index=False)
    else:
        augmented_df = pd.read_csv(f'{project_dir}/ukb_labels/ukb_labels_filecheck.csv')
    return augmented_df

def main(which_data='t1_2mm', num_epochs=50, y_variable = 'TScore', train_from_scratch = True, pretrained_model=None, new_output_dim=1, batch_size=8, val_split=0.2, test_split=0.1):
    # project_dir = '/Users/yiyuwang/Desktop/SNAPL/Projects/HEAL_prediction'
    project_dir = '/home/yiyuw/projects/MRI_TransferLearning'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Create DataFrame
    if 'ukb' in which_data:
        augmented_df = get_ukb_nifti_path(project_dir)
    else:    
        demographics = pd.read_csv(f'{project_dir}/HEAL_labels/participants_demographics.csv')
        label_df = pd.read_csv(f'{project_dir}/HEAL_labels/labels.csv')
        label_df = label_df[label_df.session == 'baseline']
        CP_df = pd.merge(label_df, demographics, on='subject_id')
        augmented_df = get_nifti_path(which_data, CP_df, project_dir)
    # Create DataLoaders for 'target_variable' column (replace with your regression target column)
    new_output_dim=1
    print('y_variable:', y_variable)
    print('new_output_dim = ', new_output_dim)
    print('****REGRESSION model! Make sure y_variable is continuous.*****')

    train_loader, val_loader, test_loader = create_dataloaders(augmented_df, y_variable, batch_size=batch_size, val_split=val_split, test_split=test_split)
    criterion = nn.MSELoss()

    # Train the model
    # if not train_from_scratch:
    #       pretrained_model = 'cnn3d_situation.pt'
        
    if train_from_scratch:
        # Initialize model, criterion, and optimizer
        output_dim = 1
        
        data_dir, data_file_name, data_shape = get_path_and_shape(which_data, project_dir)
        model = CNN3D(output_dim=output_dim, input_shape=data_shape)

        model = model.to(device)  # Move model to GPU if available
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print('train 3dcnn from scratch')
        count_params(model)
        trained_model = train_model(model, 
                                train_loader, 
                                val_loader, 
                                criterion, 
                                optimizer, 
                                num_epochs=num_epochs,
                                device=device)
        test_loss, test_r2, test_corr, test_outputs, test_labels = evaluate_model(trained_model, test_loader, criterion, device)
        
        # save model
        model_name = f'{project_dir}/models_save/cnn3d_regression_{which_data}.pt'
        torch.save(trained_model.state_dict(), model_name)

        # convert to AUC
        from sklearn.metrics import roc_auc_score
        # if test_labels > 64, then1, else 0
        test_groups = np.where(np.array(test_labels) >= 65, 1, 0)
        test_auc = roc_auc_score(test_groups, test_outputs)
        # save test_outputs, test_labels in a csv
        test_df = pd.DataFrame({'test_outputs': test_outputs, 'test_labels': test_labels})
        test_df.to_csv(f'{project_dir}/models_save/cnn3d_regression_{which_data}_test.csv', index=False)
        print(f'Test AUC: {test_auc:.4f}')

    elif pretrained_model:
        model = CNN3D(output_dim=3)
        # Dynamically compute the flattened size
        sample_shape = (1, 99, 117, 95)
        flattened_size = model._get_flattened_size(sample_shape)

        model = model.to(device)  # Move model to GPU if available
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        
        count_params(model)

        model.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
        trained_model = fine_tune_cnn3d(model, new_input_dim=flattened_size, new_output_dim=new_output_dim)
        print('load pretrained 3dcnn')
        count_params(trained_model)
        fine_tune_model = train_model(trained_model, 
                                train_loader, 
                                val_loader, 
                                criterion, 
                                optimizer, 
                                num_epochs=10,
                                device=device)
        
        evaluate_model(fine_tune_model, test_loader, criterion, device)

if __name__ == '__main__':
    main(which_data = 'ukb_t1', num_epochs=10, y_variable='BMI', batch_size=64, val_split=0.3, test_split=0.1)