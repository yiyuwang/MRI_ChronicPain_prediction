#!/bin/bash

# Yiyu Wang 2024-02-02

project_dir='/scratch/groups/smackey/HEAL'
# mkdir -p ${project_dir}/HEAL_MRI


source_dir=${project_dir}/fmriprep
target_dir=${project_dir}/HEAL_MRI



subjects_list=()
for folder in "${source_dir}"/*; do
    if [[ -d "$folder" ]]; then
        subjects_list+=("$(basename $folder)")
    fi
done
echo "subjects_list: ${subjects_list[@]}"


for subject in "${subjects_list[@]}"; do 
    mkdir -p ${target_dir}/${subject}/ses-01/anat
    file_name=${subject}_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
    T1_file_dir=${source_dir}/${subject}/ses-01/anat/${file_name}
    target_T1_dir=${target_dir}/${subject}/ses-01/anat/
    echo "cp ${T1_file_dir} ${target_T1_dir}"
    cp ${T1_file_dir} ${target_T1_dir}

    # mkdir ${target_dir}/${subject}/ses-01/func
    # file_name=${subject}_ses-01_task-pressure_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    # file_dir=${source_dir}/${subject}/ses-01/func/${file_name}
    # target_file_dir=${target_dir}/${subject}/ses-01/func/
    # cp ${file_dir} ${target_file_dir}

done




