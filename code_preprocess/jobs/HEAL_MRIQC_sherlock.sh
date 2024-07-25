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

project_dir=/scratch/groups/smackey/HEAL_MRI_BIDS
BIDS_dir=${project_dir}/fmriprep
output_dir=${project_dir}/fmriprep_output
work_dir=${project_dir}/fmriprep_work
MRIQC_dir=${project_dir}/MRIQC

subjects_list=("sub-bio0003" "sub-bio0009" "sub-bio0029" "sub-bio0033" 
"sub-bio0046" "sub-bio0056" "sub-bio0058" "sub-bio0059" "sub-bio0074" "sub-bio0088" "sub-bio0122" 
"sub-bio0194" "sub-bio0213" "sub-bio0214" "sub-bio0233" "sub-bio0234" "sub-bio0242" "sub-bio0256" 
"sub-bio0264" "sub-bio0274" "sub-bio0280" "sub-bio0286" "sub-bio0294" "sub-bio0293" "sub-bio0312")



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
for subject_num in "${subjects_list[@]}"
do
    subject=$(formatSubID $subject_num)

    if [[ -n $subject ]]; then

        echo $subject
        

        # export subject bids_anat bids_func output_anat output_func work_anat work_func 
        export subject BIDS_dir MRIQC_dir work_dir
        envsubst '${subject} ${BIDS_dir} ${MRIQC_dir} ${work_dir}' < HEAL_MRIQC_template.sbatch > scripts/HEAL_MRIQC_sub-${subject}.sbatch
    
        
        # submit them as jobs:
        if [ "$submit" -eq 1 ]; then
            # wait 30 seconds:
            sleep 15
            
            # submit the job
            echo "submit job for ${subject}"
            sbatch scripts/HEAL_MRIQC_sub-${subject}.sbatch

            # schedular to space out jobs
            job_count=$((job_count + 1))
            # check if job count exceeds 50
            if [ "$job_count" -ge 50 ]; then
                echo "Job count exceeds 50. Sleeping for 10 minutes."
                sleep 600
                job_count=0
            fi

        else
            echo "dry run for ${subject}"
        fi
    else
        echo "${fw_subject} not a subject, skip"
    fi
     


done