#!/usr/bin/env python

'''
Create Parcellations given ROIs
'''

# Yiyu Wang 2023/12/
import pandas as pd
import glob, sys
import nibabel as nib
import numpy as np
import os
from fastdtw import fastdtw

import warnings

# main
def format_sub_id(source_id):
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
        
    elif source_id.isdigit():
        sub_num = int(source_id)
        formatted_id = "bio" + "{:04}".format(sub_num)
    
    else: # everything else just return empty
        formatted_id = ""
    
    return formatted_id


mask_img = '/home/users/yiyuw/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'

ROI_scheme = str(sys.argv[1])



# function
def calculate_dtw(X):
    n_nodes = X.shape[0]
    dtw_distances = np.zeros((n_nodes, n_nodes))
    cost_matrices = {}

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            distance, path = fastdtw(X[i], X[j])
            dtw_distances[j, i] = distance
            cost_matrices[(i, j)] = path

    return dtw_distances, cost_matrices

# directories
# local directory
# project_dir = '/users/yiyuwang/Desktop/SNAPL/Projects/HEAL_prediction/'


# sherlock
project_dir = '/scratch/groups/smackey/HEAL/'
data_dir =  project_dir + f'parcellations/{ROI_scheme}'
confounds_dir =  project_dir + 'confounds/'
save_dir = project_dir + f'parcellations/{ROI_scheme}/DTW'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load all CSV files from data_dir
parcellation_files = glob.glob(data_dir + f'/*.csv')
print(f'fetch {len(parcellation_files)} files from {data_dir}')

# Iterate over each CSV file
for file in parcellation_files:
    print(file)
    s = os.path.basename(file).split('sub-')[1].split('_')[0]
    r = os.path.basename(file).split('run-')[1].split('_')[0]

    dtw_file_name = save_dir + f'/sub-{s}_run-{r}_dtw.csv'
    if os.path.exists(dtw_file_name):
        print(f'{dtw_file_name} exists, skip')
        continue
    else:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file, index_col=0)
        
        # Calculate DTW
        dtw_distances, _ = calculate_dtw(df.values)
        dtw_distances_df = pd.DataFrame(dtw_distances, index=df.index, columns=df.index)
        dtw_distances_df.to_csv(dtw_file_name, index=False, header=False)