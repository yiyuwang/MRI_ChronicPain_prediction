#!/bin/bash

# Yiyu Wang 2023-12-15


# remove incomplete fmriprep subjects

# old directory:
# BIDS_dir=/scratch/groups/smackey/P01/ScanData/fmriprep_cl
# output_dir=/scratch/groups/smackey/P01/ScanData/HEAL_fmriprep
# work_dir=/scratch/groups/smackey/P01/ScanData/HEAL_fmriprep_work

project_dir=/scratch/groups/smackey/HEAL
output_dir=${project_dir}/BIDS

not_dry_run=$1
cd $output_dir

# Iterate over each sub-folder in fmriprep
for subdir in sub-*; do
    # Check if the subdirectory follows the format "sub-XXXX" and is a directory
    if [[ -d "$subdir" && $subdir =~ ^sub-bio[0-9]{4}$ ]]; then
        # Check if "ses-01" folder exists inside the subdirectory
        if [[ -d "$subdir/ses-01" ]]; then
            # Initialize a flag to indicate the presence of required folders and files
            delete_flag=0

            # Check for the existence of "anat" and "fmap" folders inside "ses-01"
            if [[ ! -d "$subdir/ses-01/anat" || ! -d "$subdir/ses-01/fmap" || ! -d "$subdir/ses-01/func" ]]; then
                delete_flag=1
            fi

            # Check for the existence of "func" folder and required .nii.gz files inside it
            if [[ -d "$subdir/ses-01/func" ]]; then
                func_files=$(ls "$subdir/ses-01/func"/*.nii 2> /dev/null | grep "bold" || true)
                if [[ -z "$func_files" ]]; then
                    delete_flag=1
                fi
            else
                delete_flag=1
            fi

            # Delete the subdirectory and the associated HTML file if the flag is set
            if [[ $delete_flag -eq 1 ]]; then
                
                if [ "$not_dry_run" -eq 1 ]; then
                    echo "rm -rf "$subdir""
                    rm -rf "$subdir"
                else 
                    echo "dry run rm -rf "$subdir""
                fi
            fi
        else
            # "ses-01" folder does not exist, so delete the subdirectory
            if [ "$not_dry_run" -eq 1 ]; then
                echo "rm -rf "$subdir""
                rm -rf "$subdir"
            else 
                echo "dry run rm -rf "$subdir""
            fi
        fi
    fi
done
