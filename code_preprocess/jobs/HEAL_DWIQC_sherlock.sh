#!/bin/bash

# Yiyu Wang 2023-09-11

# This scripts creates sbatch script for the given subject:

# enter a subject number (use for debug and test)
# echo "Please enter a subject number:"
# read subjects_list
# echo "You entered $subjects_list"

# enter a subject list (use for batch processing):
# subjects_list=("bio0024" "bio0025" "bio0026")

which_subjects='new'
BIDS_dir=/scratch/groups/smackey/HEAL_MRI_BIDS/fmriprep
dwi_dir=/scratch/groups/smackey/HEAL_MRI_BIDS/DWI_QC
# subjects_list=($(python get_subject_folder_name.py $which_subjects $BIDS_dir $dwi_dir))
subjects_list=("sub-bio0126" "sub-bio0217" "sub-bio0193" "sub-bio0216" "sub-bio0211" "sub-bio0079" "sub-bio0169" "sub-bio0025" "sub-bio0030" "sub-bio0164" 
"sub-bio0198" "sub-bio0157" "sub-bio0103" "sub-bio0196" "sub-bio0142" "sub-bio0045" "sub-bio0041" "sub-bio0237" "sub-bio0232" "sub-bio0272" "sub-bio0268" 
"sub-bio0296" "sub-bio0090" "sub-bio0123" "sub-bio0010" "sub-bio0171" "sub-bio0270" "sub-bio0306" "sub-bio0114" "sub-bio0230" "sub-bio0020")
# subjects_list=("sub-bio0001" "sub-bio0002")
for subject in  "${subjects_list[@]}"
do
  input_dir=${BIDS_dir}/${subject}/ses-01/dwi
  output_dir=${dwi_dir}/${subject}

  export subject input_dir output_dir
  envsubst '${subject} ${input_dir} ${output_dir}' < /scripts/HEAL_dwi_template.sbatch > scripts/HEAL_dwi_${subject}.sbatch

  # wait 10second for the scripts to be created:
  sleep 30

  # submit them as jobs:
  echo "submitted job for ${subject}"
  sbatch scripts/HEAL_dwi_${subject}.sbatch
  
done


