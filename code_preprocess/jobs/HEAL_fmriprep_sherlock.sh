#!/bin/bash

# Yiyu Wang 2023-09-11

# Before running this script: 
# have a singularity for fmriprep:
# building command:
# singularity build /scratch/users/yiyuw/SingularityImages/fmriprep-23.1.2.simg docker://nipreps/fmriprep:23.1.2

# For testing:
echo "Please enter a subject_number:"
read subjects_list
echo "You entered $subjects_list"

# for batch processing:
# to do: run a script that auto-check the participants to be run:



# 1 => submit them as jobs
# 0 > do not submit them as jobs
submit=$1

# 1 -> run debug version
# 0 -> not debug version
# debug=$2

# old directory:
# BIDS_dir=/scratch/groups/smackey/P01/ScanData/fmriprep_cl
# output_dir=/scratch/groups/smackey/P01/ScanData/HEAL_fmriprep
# work_dir=/scratch/groups/smackey/P01/ScanData/HEAL_fmriprep_work


project_dir=/scratch/groups/smackey/HEAL_MRI_BIDS
BIDS_dir=${project_dir}/fmriprep
output_dir=${project_dir}/fmriprep_output
work_dir=${project_dir}/fmriprep_work
code_dir=${project_dir}/code/jobs


formatSubID() {
    source_id=$1
    formatted_id=""

    # Check if source_id contains "bio" or "Bio"
    if [[ $source_id == *"bio"* ]] || [[ $source_id == *"Bio"* ]] || [[ $source_id == *"BIO"* ]] || [[ $source_id == *"sub"* ]]; then
        # Extract numbers
        sub_num=$(echo $source_id | grep -o -E '[0-9]+' | sed 's/^0*//')
        
        # Format the sub number with leading zeros to make it four digits
        formatted_id="bio$(printf '%04d' $sub_num)"

    elif [[ $source_id =~ ^0*[0-9]+$ ]]; then
        # If source_id is just numbers, remove leading zeros for decimal interpretation
        decimal_id=$((10#$source_id))
        formatted_id="bio$(printf '%04d' $decimal_id)"
    fi

    echo $formatted_id
}

job_count=0

for subject_num in "${subjects_list[@]}"
do
    subject=$(formatSubID $subject_num)

    if [[ -n $subject ]]; then

        echo $subject
        
        export subject BIDS_dir output_dir work_dir code_dir project_dir

        envsubst '${subject} ${BIDS_dir} ${output_dir} ${work_dir} ${code_dir} ${project_dir}' < HEAL_fmriprep_template.sbatch > scripts/HEAL_fmriprep_sub-${subject}.sbatch
    

        # submit them as jobs:
        if [ "$submit" -eq 1 ]; then
            echo "submit job for ${subject}"
            
            # wait 45 seconds for the scripts to be created and
            # spacing out jobs so that they don't overwhelm the scheduler and disk quota
            sleep 5s
            
            sbatch scripts/HEAL_fmriprep_sub-${subject}.sbatch
            
            job_count=$((job_count + 1))
            
    
        else
            echo "dry run for ${subject}"
        fi
    else
        echo "${subject_num} not a subject, skip"
    fi
done


