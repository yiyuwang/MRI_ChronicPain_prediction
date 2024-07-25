import os
import json
import nibabel as nib
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import random
import numpy as np
class MultiMRIDataset(Dataset):
    def __init__(self, root_dir, transform, data_df, label='sex', label_smoothing=False):
        self.root_dir = root_dir

        # get label_df
        self.label_df = data_df
        self.patients = [eid for eid in self.label_df['eid']]
    
        # self.patients = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.transform = transform
        self.label_smoothing = label_smoothing
        if self.label_smoothing:
            assert label == 'age' # implemented for age
            self.y_transform = LabelSmoothing() 
        # load labels:
        assert label in ['sex', 'age', 'group']
        self.label = label
        
        self.label_df = self.label_df.set_index('eid') # set patient id as index
        # self.label_df['group_name'] = self.label_df['group']
        # Convert string labels to integer class labels
        # self.label_df['group'] = pd.factorize(self.label_df['group_name'])[0]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # get data: 
        file_path = os.path.join(self.root_dir, self.label_df.loc[self.patients[idx]]['path'])

        mri_data = nib.load(file_path).get_fdata()
        mri_tensor = torch.from_numpy(mri_data).float()

        if self.transform:
            mri_tensor = self.transform(mri_tensor)

        # get label:
        pat_id = self.patients[idx]
        y = self.label_df.loc[pat_id][self.label] 
        if self.label_smoothing:
            y = self.y_transform(y)
        return mri_tensor, y, pat_id



class HEALMRIDataset(Dataset):
    def __init__(self, root_dir, session='baseline', transform=None, label='sex', label_smoothing=False):
        self.root_dir = root_dir
        self.transform = transform
        self.label_smoothing = label_smoothing
        if self.label_smoothing:
            assert label == 'age' # implemented for age
            self.y_transform = LabelSmoothing() 
        # load labels:
        assert label in ['sex', 'age', 'group', 'PainInT']
        self.label = label
        self.label_df = pd.read_csv(
                os.path.join(root_dir, 
                             f'{session}_labels_data.csv'
                )
        )
        self.label_df['group_name'] = self.label_df['group']
        # Convert string labels to integer class labels
        # self.label_df['group'] = pd.factorize(self.label_df['group_name'])[0]
        
        # load patients id based on the labels_df and has fMRI data:
        
        self.patients = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d in self.label_df['eid'].values])
        self.label_df = self.label_df.set_index('eid') # set patient id as index
       
    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        # get data: 
        patient_dir = os.path.join(self.root_dir, self.patients[idx])
        file_path = os.path.join(patient_dir, "ses-01/anat", f"{self.patients[idx]}_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
        mri_data = nib.load(file_path).get_fdata()
        mri_tensor = torch.from_numpy(mri_data).float()

        if self.transform:
            mri_tensor = self.transform(mri_tensor)

        # get label:
        pat_id = self.patients[idx]
        y = self.label_df.loc[pat_id][self.label] 
        if self.label_smoothing:
            y = self.y_transform(y)
        return mri_tensor, y
    
    

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, label='sex', label_smoothing=False):
        self.root_dir = root_dir
        self.patients = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.transform = transform
        self.label_smoothing = label_smoothing
        if self.label_smoothing:
            assert label == 'age' # implemented for age
            self.y_transform = LabelSmoothing() 
        # load labels:
        assert label in ['sex', 'BMI', 'age', 'group']
        self.label = label
        self.label_df = pd.read_csv(
                os.path.join(root_dir, 
                             'uk_biobank_chronic_low_back_pain_data.csv'
                )
        )
        self.label_df = self.label_df.set_index('eid') # set patient id as index
        self.label_df['group_name'] = self.label_df['group']
        # Convert string labels to integer class labels
        self.label_df['group'] = pd.factorize(self.label_df['group_name'])[0]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # get data: 
        patient_dir = os.path.join(self.root_dir, self.patients[idx])
        file_path = os.path.join(patient_dir, "ses-2_0/anat", f"{self.patients[idx]}_ses-2_0_desc-preproc_T1w_brain_space-1mm_MNI.nii.gz")
        mri_data = nib.load(file_path).get_fdata()
        mri_tensor = torch.from_numpy(mri_data).float()

        if self.transform:
            mri_tensor = self.transform(mri_tensor)

        # get label:
        pat_id = int(self.patients[idx].split('-')[1]) # convert sub-123 to 123
        y = self.label_df.loc[pat_id][self.label] 
        if self.label_smoothing:
            y = self.y_transform(y)
        return mri_tensor, y

def get_subset_indices(split_name, split_dict, dataset):
    """
    Returns the indices of the dataset based on the IDs provided in the split.
    """
    ids = split_dict[split_name]

    return [dataset.patients.index(str(id_))  for id_ in ids]

class ZScoreNormalize:
    def __init__(self, mean=241.0575, std=419.5543, epsilon=1e-10): 
        """
        mean, std: precomputed mean and standard deviation from the training dataset.
        epsilon: a small constant added to the standard deviation for numerical stability.
        --> precomputed mean = 241.0575, std = 419.5543 
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, tensor):
        """
        Apply Z-score normalization to the tensor.
        
        tensor: PyTorch tensor of any shape.
        """
        return (tensor - self.mean) / (self.std + self.epsilon)

class RandomSlicesTransform:
    def __init__(self, num_slices=10):
        self.num_slices = num_slices

    def __call__(self, tensor):
        """
        Assumes tensor is of shape (D, H, W) where D is the depth (number of slices).
        Returns a tensor of shape (num_slices, H, W).
        """
        D, H, W = tensor.shape
        if D <= self.num_slices:
            return tensor

        # Randomly select indices for slices without replacement
        selected_slices = random.sample(range(D), self.num_slices)
        return tensor[selected_slices, :, :]

class EvenlySpreadSlices:
    def __init__(self, total_slices=182, desired_slices=32, exclude_slices=10):
        self.total_slices = total_slices - 2 * exclude_slices  # adjust for excluded slices
        self.desired_slices = desired_slices
        self.interval = self.total_slices / self.desired_slices
        self.exclude_slices = exclude_slices

    def __call__(self, volume):
        """
        :param volume: a 3D tensor representing the MRI volume with shape [Depth, Height, Width]
        :return: a 3D tensor with `self.desired_slices` slices from the input volume.
        """
        
        # Calculate indices of slices to select, considering the excluded slices
        indices = [int(i * self.interval) + self.exclude_slices for i in range(self.desired_slices)]
        
        # Select slices based on the calculated indices
        selected_slices = volume[indices]

        return selected_slices

class NormalizeTo1:
    def __init__(self, min_val=-55.1429, max_val=3241.3420):
        self.min_val = min_val
        self.max_val = max_val
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        normalized_image = (image - self.min_val) / (self.max_val - self.min_val)
        return normalized_image 

class RepeatChannelTransform:
    def __init__(self, num_channels=3):
        self.num_channels = num_channels

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where C = 1.

        Returns:
            Tensor: Repeated tensor of size (num_channels, H, W).
        """
        x = tensor.unsqueeze(1) 
        x = x.repeat(1, self.num_channels, 1, 1)
        #print(f'After repeat: {x.shape=}')
        return x

def get_datasets(dataset, split_dict):
    """ Util function to return splits"""
    # Let's assume you want to use the training set. If you want to use 'test' or 'val' set, just change the split_name
    datasets = []
    for split_name in ['train_ids', 'val_ids', 'test_ids']:
        indices = get_subset_indices(split_name, split_dict, dataset)
        current_dataset = Subset(dataset, indices)
        datasets.append(current_dataset)
    return datasets

class AgeToBin:
    def __init__(self, min_age=20, max_age=90):
        self.min_age = min_age
        self.max_age = max_age
        self.n_bins = max_age - min_age + 1

    def __call__(self, age):
        age = age.item()  # if age is a tensor
        if age < self.min_age:
            return 0
        elif age > self.max_age:
            return self.n_bins - 1 #self.max_age - self.min_age
        else:
            return age - self.min_age

class LabelSmoothing:
    def __init__(self, min_age=20, max_age=90, sigma=5):
        self.min_age = min_age
        self.max_age = max_age
        self.sigma = sigma
        self.n_bins = max_age - min_age + 1
        self.bins = torch.arange(self.n_bins)
        self.age_to_bin = AgeToBin(min_age, max_age)

    def __call__(self, age):
        bin_index = self.age_to_bin(age)
        smoothed_labels = torch.exp(-(self.bins - bin_index)**2 / (2 * self.sigma**2))
        return smoothed_labels
        #return F.softmax(smoothed_labels, dim=-1)

if __name__ == '__main__':

    root_path = "data/HEAL_MRI"

    # Create composed transforms
    composed_transforms = T.Compose([
        #ZScoreNormalize(),
        T.Resize((224,224))
    ])

    dataset = HEALMRIDataset(root_path, transform=composed_transforms, label='age', label_smoothing=True)
    # from IPython import embed; embed()

    # Load the splits from the json file
    with open('data/HEAL_MRI/baseline_split.json', 'r') as f:
        split_dict = json.load(f)

    # Let's assume you want to use the training set. If you want to use 'test' or 'val' set, just change the split_name
    split_name = 'train_ids'  # or 'val_ids' or 'test_ids'
    indices = get_subset_indices(split_name, split_dict, dataset)

    current_dataset = Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(current_dataset, batch_size=4, shuffle=True if split_name == 'train_ids' else False)
   
    # # Compute stats for Z-score:
    all_train_data = torch.cat([data for data, _ in tqdm(loader)], dim=0)  
    #from IPython import embed; embed()
    d_min = all_train_data.min()
    d_max = all_train_data.max()
    print(f'{d_min}, {d_max}')
    #import sys; sys.exit() 

    import timm
    model = timm.create_model('swin_large_patch4_window7_224.ms_in22k', pretrained=True)
    num_custom_classes = 2 
    model.head = torch.nn.Linear(model.head.in_features, num_custom_classes)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    # from IPython import embed; embed()
 
    # Let's iterate over the batches and print the shape of the data
    for batch in loader:
        print(batch[0].shape) 
        print((batch[0]).mean())
        print((batch[0]).std())

        # from IPython import embed; embed()    

        img = batch[0]
        # collapsing depth and batch dimension
        i = img.view(-1, 224, 224).unsqueeze(1)
        # repeat channel dim (3 expected)
        rgb_like_tensor = i.repeat(1, 3, 1, 1)
        model(rgb_like_tensor) 

        from explore import MRI_Viewer
        viewer = MRI_Viewer(load_tensor=img[0])
        plt.show()


        break
