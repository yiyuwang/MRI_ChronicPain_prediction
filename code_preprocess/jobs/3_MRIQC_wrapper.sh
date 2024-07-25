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
output_dir=${project_dir}/fmriprep
work_dir=${project_dir}/fmriprep_work
code_dir=${project_dir}/code/jobs
MRIQC_dir=${project_dir}/MRIQC

# subjects_list=($(python get_subject_folder_name.py $which_subjects $BIDS_dir $output_dir))
subjects_list=("sub-bio0003" "sub-bio0009" "sub-bio0029" "sub-bio0033" 
"sub-bio0046" "sub-bio0056" "sub-bio0058" "sub-bio0059" "sub-bio0074" "sub-bio0088" "sub-bio0122" 
"sub-bio0194" "sub-bio0213" "sub-bio0214" "sub-bio0233" "sub-bio0234" "sub-bio0242" "sub-bio0256" 
"sub-bio0264" "sub-bio0274" "sub-bio0280" "sub-bio0286" "sub-bio0294" "sub-bio0293" "sub-bio0312")


# "sub-bio0049" "sub-bio0311" "sub-bio0173" "sub-bio0254" "sub-bio0134" "sub-bio0278" 
# "sub-bio0024" "sub-bio0139" "sub-bio0290" "sub-bio0235" "sub-bio0141" "sub-bio0206" "sub-bio0014" 
# "sub-bio0077" "sub-bio0209" "sub-bio0063" "sub-bio0001" "sub-bio0324" "sub-bio0291" "sub-bio0236" 
# "sub-bio0027" "sub-bio0099" "sub-bio0116" "sub-bio0197" "sub-bio0276" "sub-bio0297" "sub-bio0189" 
# "sub-bio0104" "sub-bio0328" "sub-bio0026" "sub-bio0260" "sub-bio0225" "sub-bio0018" "sub-bio0153" 
# "sub-bio0113" "sub-bio0325" "sub-bio0064" "sub-bio0098" "sub-bio0159" "sub-bio0071" "sub-bio0152" 
# "sub-bio0238" "sub-bio0021" "sub-bio0247" "sub-bio0204" "sub-bio0085" "sub-bio0183" "sub-bio0070" 
# "sub-bio0191" "sub-bio0318" "sub-bio0300" "sub-bio0303" "sub-bio0329" "sub-bio0072" "sub-bio0081" 
# "sub-bio0224" "sub-bio0295" "sub-bio0002" "sub-bio0273" "sub-bio0243" "sub-bio0117" "sub-bio0210" 
# "sub-bio0161" "sub-bio0015" "sub-bio0102" "sub-bio0313" "sub-bio0201" "sub-bio0091" "sub-bio0112" 
# "sub-bio0338" "sub-bio0284" "sub-bio0083" "sub-bio0066" "sub-bio0073" "sub-bio0309" "sub-bio0299" 
# "sub-bio0271" "sub-bio0051" "sub-bio0055" "sub-bio0031" "sub-bio0227" "sub-bio0212" "sub-bio0075" 
# "sub-bio0054" "sub-bio0220" "sub-bio0039" "sub-bio0244" "sub-bio0078" "sub-bio0022" "sub-bio0124" 
# "sub-bio0239" "sub-bio0144" "sub-bio0248" "sub-bio0289" "sub-bio0249" "sub-bio0319" "sub-bio0240" 
# "sub-bio0265" "sub-bio0111" "sub-bio0199" "sub-bio0283" "sub-bio0092" "sub-bio1111" "sub-bio0258" 
# "sub-bio0287" "sub-bio0251" "sub-bio0314" "sub-bio0006" "sub-bio0120" "sub-bio0163" "sub-bio0110" 
# "sub-bio0261" "sub-bio0065" "sub-bio0155" "sub-bio0223" "sub-bio0080" "sub-bio0285" "sub-bio0032" 
# "sub-bio0222" "sub-bio0096" "sub-bio0019" "sub-bio0298" "sub-bio0252" "sub-bio0125" "sub-bio0162" 
# "sub-bio0288" "sub-bio0089" "sub-bio0060" "sub-bio0170" "sub-bio0115" "sub-bio0221" "sub-bio0035" 
# "sub-bio0267" "sub-bio0259" "sub-bio0126" "sub-bio0217" "sub-bio0193" "sub-bio0216" "sub-bio0211" 
# "sub-bio0079" "sub-bio0169" "sub-bio0025" "sub-bio0030" "sub-bio0164" "sub-bio0198" "sub-bio0157" 
# "sub-bio0103" "sub-bio0196" "sub-bio0142" "sub-bio0045" "sub-bio0041" "sub-bio0237" "sub-bio0232" 
# "sub-bio0272" "sub-bio0268" "sub-bio0296" "sub-bio0090" "sub-bio0123" "sub-bio0010" "sub-bio0171" 
# "sub-bio0270" "sub-bio0306" "sub-bio0114" 

# "bio0001" "bio0002" "bio0003" "bio0006" "bio0032" "bio0007"
# "bio0096" "bio0126" "bio0177" "bio0214" "bio0239" "bio0268" "bio0293" "bio0320"
# "bio0033" "bio0071" "bio0098" "bio0134" "bio0183" "bio0216" "bio0240" "bio0294" "bio0324" "bio0035" "bio0072"  
# "bio0099" "bio0139" "bio0189" "bio0217" "bio0242" "bio0270" "bio0295" "bio0325" "bio0039" "bio0073" "bio0102"  
# "bio0141" "bio0191" "bio0218" "bio0243" "bio0271" "bio0296" "bio0328" "bio0009" "bio0041" "bio0074" 
# "bio0103" "bio0142" "bio0193" "bio0220" "bio0244" "bio0272" "bio0297" "bio0329")


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