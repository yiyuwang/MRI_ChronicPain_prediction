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

from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker
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

task = sys.argv[1]
print(f'parcellating task - {task}')
ROI_scheme = sys.argv[2]
if ROI_scheme == 'schaefer':
    NROI = int(sys.argv[3]) # NROI will not be used if ROI_scheme is msdl
    ROI_scheme = f'schaefer{NROI}'
print(f'using ROI scheme - {ROI_scheme}')

if ROI_scheme == 'msdl':
    atlas = datasets.fetch_atlas_msdl()
    masker = NiftiMapsMasker(
        maps_img=atlas['maps'],
        resampling_target="mask",
        t_r=0.8,
        detrend=True,
        memory='/scratch/users/yiyuw/nilearn',
        mask_img = mask_img,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        verbose=0
    )
elif 'schaefer' in ROI_scheme:
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=NROI,resolution_mm=2)
    masker = NiftiLabelsMasker(
        labels_img=atlas['maps'],
        labels=atlas['labels'],
        t_r = 0.8,
        detrend=True,
        standardize=True,
        memory='/scratch/users/yiyuw/nilearn',
        mask_img = mask_img,
        verbose=0)

# names of the core regs I want (outliers added by subject as varies)
tsv_regs = ['framewise_displacement',
            'trans_x','trans_x_derivative1',
            'trans_y','trans_y_derivative1',
            'trans_z','trans_z_derivative1',
            'rot_x','rot_x_derivative1',
            'rot_y','rot_y_derivative1',
            'rot_z','rot_z_derivative1',
            'a_comp_cor_00','a_comp_cor_01','a_comp_cor_02',
            'a_comp_cor_03','a_comp_cor_04','a_comp_cor_05']

# directories
project_dir = '/scratch/groups/smackey/HEAL/'
data_dir = project_dir + 'fmriprep/'
save_dir =  project_dir + 'parcellations/'
confounds_dir =  project_dir + 'confounds/'

if task == 'rest':
    tsv_files = glob.glob(data_dir + f"**/**/func/*task-{task}_run-[1|2]*.tsv")
elif task == 'pressure':
    tsv_files = glob.glob(data_dir + f"**/**/func/*task-{task}*.tsv")
else:
    raise Exception('task not recognized')
    
print(f'total of {len(tsv_files)} tsc files fetched')


# create confounds files
mask_file = '/home/users/yiyuw/masks/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii'

# select tsv files for sub-bio269 and sub-bio0301
tsv_files = [file for file in tsv_files if 'sub-bio0269' in file or 'sub-bio0301' in file]

for file in tsv_files:
    s = os.path.basename(file).split('sub-')[1].split('_ses')[0]  
    print(s)
    if task == 'rest':
        run = os.path.basename(file).split('run-')[1].split('_desc')[0]
    elif task == 'pressure':
        run = '1'    
    # print(f'\trunning sub-{s}, run-{run}')
    tsv_data = pd.read_csv(file, delimiter='\t')

    filename = confounds_dir + f'/sub-{s}_task-{task}_run-{run}_confounds.csv'
    if os.path.exists(filename): # if file does exist, skip
        continue
    else: # if file does not exist, create it
        # Check if all columns in tsv_regs exist in tsv_data
        missing_cols = [col for col in tsv_regs if col not in tsv_data.columns]
        if missing_cols:
            print("Missing columns in tsv_data:", missing_cols)
        else: # all columns exist
            confounds = tsv_data[tsv_regs] 
            #add outliers
            outlier_regs = [i for i in tsv_data.columns.tolist() if 'outlier' in i] 
            confounds = pd.concat([confounds, tsv_data[outlier_regs]], axis=1)
            #replace NaNs with 0s
            confounds.values[np.isnan(confounds.values)]=0
            #save as csv
            confounds.to_csv(filename, index=False) 

# Run Parcellation
confounds_list = glob.glob(confounds_dir + f"*task-{task}*csv")

# create save_dir 
if not os.path.exists(save_dir + f'{ROI_scheme}'):
    print(f'creating folder {save_dir}/{ROI_scheme}')
    os.mkdir(save_dir + f'{ROI_scheme}')

confoulds_list = [file for file in confounds_list if 'sub-bio0269' in file or 'sub-bio0301' in file]

for file in confounds_list:
    s = os.path.basename(file).split('sub-')[1].split('_task')[0]    
    run = os.path.basename(file).split('run-')[1].split('_confounds')[0]
    if task == 'rest':
        func_path = data_dir + f"sub-{s}/**/func/*task-{task}_run-{run}*space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    elif task == 'pressure':
        func_path = data_dir + f"sub-{s}/**/func/*task-{task}*space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    else:
        raise Exception('task not recognized')    
    print(func_path)
    
    if glob.glob(func_path):
        save_file_path = save_dir + f'{ROI_scheme}/sub-{s}_task-{task}_run-{run}_parcellation-{ROI_scheme}.csv'
        
        if len(glob.glob(save_file_path))== 0: # did not already calculate
            cov = pd.read_csv(file)
            cov.values[np.isnan(cov.values)]=0
            
            print(save_file_path)

            # load fmri img
            fmri_img = nib.load(glob.glob(func_path)[0])

            # clean_img:
            # clean_img = clean_img(fmri_img, confounds=cov, detrend=True, standardize=True, low_pass=0.1, high_pass=0.01, t_r=0.8)

            # parcellate and denoise with confounds
            parcellated_data = masker.fit_transform(fmri_img,confounds=cov)

            # save parcellation to df
            parcellated_data_df = pd.DataFrame(parcellated_data,columns=atlas['labels'])

            # save df 
            parcellated_data_df.to_csv(save_file_path)  

        else: # alraedy calculated
            continue
    else:
        
        print("\tfunc file does not exist!")
        continue
  



