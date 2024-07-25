#!/usr/bin/env python

'''
downsample ukb T1 from 1mm to 2mm
'''

# convert 1mm to 2mm space:


import numpy as np
import glob
import nibabel as nib
import os
from nilearn.image import resample_img
target_affine = np.diag((2, 2, 2))

ukb_files = glob.glob('/home/yiyuw/projects/MRI_TransferLearning/uk_biobank/sub-*/ses-2_0/anat/sub-*_ses-2_0_desc-preproc_T1w_brain_space-1mm_MNI.nii.gz')


for f in ukb_files:
    print(f)
    img = nib.load(f)
    new_img = resample_img(img, target_affine=target_affine)
    new_img.to_filename(f.replace('space-1mm_MNI', 'space-2mm_MNI'))