#!/usr/bin/env python

'''
Create Parcellations given ROIs
'''

# Yiyu Wang 2023/12/
import os

import glob
import nibabel as nib
import numpy as np
import pandas as pd
import copy

import nilearn
from nilearn import datasets
from nilearn.image import smooth_img, resample_to_img
from nilearn import plotting
from nilearn.plotting import plot_glass_brain, plot_stat_map, plot_design_matrix
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from scipy.stats import norm


import gzip

import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join

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

# function
def AddSteadyStateOutliers(columns_of_interest, all_columns):
    new_columns = copy.deepcopy(columns_of_interest)
    for column in all_columns:
        if 'Outlier' in column:
            new_columns.append(column)
            
    return new_columns

def CreateConfoundMatrix(confounds_of_interest, s, fmriprep_dir):  
    confounds = pd.read_csv(join(fmriprep_dir, GetConfoundsPath(s)), sep='\t')
    confounds_of_interest = AddSteadyStateOutliers(confounds_of_interest, confounds.columns)
    cov = confounds[confounds_of_interest] 
        
    cov.values[np.isnan(cov.values)]=0
    return cov


def GetFuncFileName(sub):
    FILE_NAME = f"sub-{sub}/ses-01/func/sub-{sub}_ses-01_task-pressure_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    return FILE_NAME


def GetConfoundsPath(sub_id):
    confounds_path =  f"sub-{sub_id}/ses-01/func/sub-{sub_id}_ses-01_task-pressure_desc-confounds_timeseries.tsv"
    return confounds_path


# if __name__ == '__main__':
# directories
# local directory
# project_dir = '/users/yiyuwang/Desktop/SNAPL/Projects/HEAL_prediction/'

# sherlock
project_dir = '/scratch/groups/smackey/HEAL/'
events_dir = project_dir + 'pressure_task_event_files/'
fmriprep_dir = project_dir + 'fmriprep/'
output_dir = project_dir + 'fmri_results/univariate/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

confounds_of_interest = ['csf',
                        'white_matter',
                        'trans_x', 
                        'trans_y', 
                        'trans_z',
                        'rot_x',
                        'rot_y',
                        'rot_z','framewise_displacement']


mask_img = '/home/users/yiyuw/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'

gm_mask_img = datasets.load_mni152_gm_mask(resolution=2, threshold=0.2, n_iter=2)

TR = 0.8
n_scans = 477
frame_times = np.arange(n_scans) * TR 

# check the participants who are ready to run:
# create subject_list based on the subjects who have the eprime files:
eprime_files = glob.glob(f'{events_dir}/*_task-pressure_events.csv')
eprime_subjects = []
for file in eprime_files:
    
    sub_id = file.split("sub-")[1].split("_task")[0]
    formatted_id = format_sub_id(sub_id)
    eprime_subjects.append(formatted_id)

print(eprime_subjects)
can_run_list = []
# double check that they also have fmriprep files:
for sub in eprime_subjects:
    if not os.path.isfile(fmriprep_dir +  GetFuncFileName(sub)):
        print(f"{sub} does not have fmriprep files")
    else:
        can_run_list.append(sub)
        
print(f"{len(can_run_list)} participants are ready")


# model1: pain stimulation main effect:
dm_name = 'model1_stimulation'
dm_dir = output_dir + f'{dm_name}/'


if not os.path.isdir(dm_dir):
    os.mkdir(dm_dir)

res_dir = dm_dir + f'1stlvl'
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

for sub in can_run_list:

    print(f'running subject {sub}')
    sub_output_dir = res_dir + f'/{sub}/'
    if not os.path.isdir(sub_output_dir):
        os.mkdir(sub_output_dir)

    all_events_df = pd.read_csv(f'{events_dir}/sub-{sub}_task-pressure_events.csv')

    # Replace 'pain' containing values with 'stimulation'
    all_events_df.loc[all_events_df['trial_type'].str.contains('pain'), 'trial_type'] = 'stimulation'

    # Create a new dataframe with zeros
    events = pd.DataFrame(0, index=np.arange(477/TR), columns=all_events_df['trial_type'].unique())

    # Update the new dataframe based on the original dataframe
    for idx, row in all_events_df.iterrows():
        start = int(row['onset'])
        end = int(row['onset'] + row['duration'])
        events.loc[start:end, row['trial_type']] = 1


    #get confounds info:
    cov = CreateConfoundMatrix(confounds_of_interest, sub, fmriprep_dir)

    # Construct first level model parameters
    fmri_glm = FirstLevelModel(t_r=TR,
                        noise_model='ar3',
                        standardize=True,
                        hrf_model='spm',
                        drift_model='cosine',
                        high_pass=.01, mask_img=gm_mask_img,smoothing_fwhm=6)

    # load func image
    func_path = fmriprep_dir +  GetFuncFileName(sub)
    func_img = nib.load(func_path)

    # fit first level glm
    fmri_glm = fmri_glm.fit(func_img, all_events_df, confounds=cov)

    # save design_matrix for every run
    design_matrix = fmri_glm.design_matrices_[0]

    # save design matrix image for future inspection
    plot_design_matrix(design_matrix, output_file=join(sub_output_dir, f'design_matrix.png'))
    contrast_matrix = np.eye(design_matrix.shape[1])

    # extract and save the betas
    column_names = ["stimulation", "ready", "rating"]
    column_indices = [ design_matrix.columns.get_loc(col) for col in column_names]
    for ci in column_indices:
        print(f'saving regressor for regressor {design_matrix.columns[ci]}')
        eff = fmri_glm.compute_contrast(contrast_matrix[ci],output_type='stat')
        nii_file_path = sub_output_dir + f'/sub-{sub}_stat-beta_reg-{design_matrix.columns[ci]}_gm_masked.nii.gz'
        nib.save(eff, nii_file_path)
        
        eff = fmri_glm.compute_contrast(contrast_matrix[ci],output_type='z_score')
        nii_file_path = sub_output_dir + f'/sub-{sub}_stat-z_reg-{design_matrix.columns[ci]}_gm_masked.nii.gz'
        nib.save(eff, nii_file_path)


