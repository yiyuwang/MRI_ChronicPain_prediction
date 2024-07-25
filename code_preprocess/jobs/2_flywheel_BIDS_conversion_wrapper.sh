#!/bin/bash
#
# Yiyu Wang 2023-11-11

# run BIDS conversion for all dicom files downloaded from flywheel 
# adopted from script from Christine's create_fmriprep.sh
# modified by Yiyu Wang, 2023/12/12


# take the following arguments:
# BIDS_dir
# dicoms_dir
# 

#which_subjects='all'
which_subjects='new'
project_dir=/scratch/groups/smackey/HEAL

BIDS_dir=${project_dir}/BIDS
fw_dir=${project_dir}/SUBJECTS
# subjects_list=($(python get_subject_folder_name.py $which_subjects $fw_dir $BIDS_dir))
subjects_list=("bio0301" "bio0269")


# 1 => submit them as jobs
# 0 > do not submit them as jobs
submit=$1


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
for fw_subject in  "${subjects_list[@]}" # need to change this subject ID
do
    fw_subject=${fw_subject//\"/}
    subject=$(formatSubID $fw_subject)
        if [[ -n $subject ]]; then
            echo $fw_subject
            echo $subject
            export fw_subject subject
        envsubst '${fw_subject} ${subject}' < HEAL_flywheel_BIDS_template.sbatch > scripts/HEAL_BIDS_sub-${subject}.sbatch
        
        # submit them as jobs:
        if [ "$submit" -eq 1 ]; then
            # wait 30 seconds:
            sleep 30
            
            # submit the job
            echo "submit job for ${subject}"
            sbatch scripts/HEAL_BIDS_sub-${subject}.sbatch

            # schedular to space out jobs
            job_count=$((job_count + 1))
            # check if job count exceeds 50
            if [ "$job_count" -ge 50 ]; then
                echo "Job count exceeds 50. Sleeping for 20 minutes."
                sleep 1200
                job_count=0
            fi

        else
            echo "dry run for ${subject}"
        fi
    else
        echo "${fw_subject} not a subject, skip"
    fi
     


done