#!/bin/bash

# Yiyu Wang 2023-12-13

# download flywheel project
# export the list of subjects
# fw cli: if run directly through an interactive node.
# fw login lucascenter.flywheel.io:djEsnk9XPNPXhjPPCe-ecXc2tfCCE0r1j0ZKrFYR3Nsl8qjR9EwDfvIbw
# fw sync --include dicom fw://smackey/HEAL /scratch/users/yiyuw/ -z --tmp-path /scratch/users/yiyuw/HEAL/temp

project_dir='/scratch/users/yiyuw/HEAL/'

# fw cli 
# make sure to request a computing node using "sh_dev" 
# (for first sync, you might need to request a longer node than 1hr which is the default, and usually one hour is enough for updating)
# fw login lucascenter.flywheel.io:djEsnk9XPNPXhjPPCe-ecXc2tfCCE0r1j0ZKrFYR3Nsl8qjR9EwDfvIbw
# fw sync --include dicom fw://smackey/HEAL /scratch/users/yiyuw/ -z --tmp-path /scratch/users/yiyuw/HEAL/temp

# if run in the script, uncomment these two lines
# fw login lucascenter.flywheel.io:djEsnk9XPNPXhjPPCe-ecXc2tfCCE0r1j0ZKrFYR3Nsl8qjR9EwDfvIbw
# fw sync --include dicom fw://smackey/HEAL /scratch/users/yiyuw/ -z --tmp-path /scratch/users/yiyuw/HEAL/temp
# fw sync --include dicom fw://smackey/HEAL /scratch/groups/smackey/ -z --tmp-path /scratch/groups/smackey/HEAL/temp


# extract subject_list to run
# which_subjects='all'
which_subjects='new'

BIDS_dir=${project_dir}/BIDS
fw_dir=${project_dir}/SUBJECTS
subjects_list=$(python get_subject_folder_name.py $which_subjects $fw_dir $BIDS_dir)

# pass the subjects_list to 2_flywheel_BIDS_conversion_wrapper.sh

# 1 => submit them as jobs
# 0 > do not submit them as jobs
submit=1

sh 2_flywheel_BIDS_conversion_wrapper.sh $subjects_list $submit
