#!/usr/bin/bash
#SBATCH --job-name=${subject}_BIDS
#SBATCH --output=output/${subject}_BIDS_%A.out 
#SBATCH --error=output/${subject}_BIDS_%A.err
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
start=`date +%s`

# run BIDS conversion for all dicom files downloaded from flywheel 
# adopted from script from Christine's create_fmriprep.sh
# modified by Yiyu Wang, 2023/12/13:


# take the following arguments:
# echo $fw_subject


# sync the project first: 
# fw sync --include dicom -z --tmp-path '/scratch/users/yiyuw/flywheel/temp/' fw://smackey/HEAL /scratch/users/yiyuw/projects/
source activate /scratch/users/yiyuw/MyCondaEnvs/prediction_env
project_dir=/scratch/groups/smackey/HEAL
session='01'
# directory to save the BIDS
BIDS_dir=${project_dir}/BIDS

# directory where to sync project with flywheel
fw_dir=${project_dir}/SUBJECTS

# directory where the dicom files are extracted to
dicom_dir=${project_dir}/DICOMS
mkdir -p "${dicom_dir}/${subject}/"

mkdir -p ${BIDS_dir}/sub-${subject}/ses-${session}
mkdir -p ${BIDS_dir}/sub-${subject}/ses-${session}/anat
mkdir -p ${BIDS_dir}/sub-${subject}/ses-${session}/dwi
mkdir -p ${BIDS_dir}/sub-${subject}/ses-${session}/fmap
mkdir -p ${BIDS_dir}/sub-${subject}/ses-${session}/func



# 1. if subject is downloaded from website, unzip the tar file and save to the directory subejct name:
# mkdir ${fw_dir}/${subject}
# tar -xf ${fw_dir}/${fw_subject}.tar -C ${fw_dir}/${subject}

# 2.move the relevant files to the correct level: 
# Define the list of folder names to search for
list_to_extract=("ORIG 3D Ax T1 BRAVO" "ORIG 3D Ax T2 Cube" "ORIG_ DWI" "ORIG_DWI PEpolar change CV" "Pressure fMRI" "Pressure fMRI PEpolar" "Rest1" "Rest1 PEpolar" "Rest2" "Rest2 PEpolar")


# example file path:
# /scratch/users/yiyuw/flywheel/HEAL/SUBJECTS/BIO0267/*SESSIONS/*BRAIN/*ACQUISITIONS/Rest1/FILES

# iterate through the list of folder names
for folder_name in "${list_to_extract[@]}"; do
    echo "Searching for folders named '$folder_name' in ${fw_dir}/$fw_subject"
    
    # Use find command to search for folders matching the name
    find ${fw_dir}/$fw_subject/ -type d -name "${folder_name}" 2>/dev/null | while read -r found_dir; do
    
    mutli_dir=$(find ${fw_dir}/$fw_subject/ -type d -name "${folder_name}_1")
    if [[ -n $mutli_dir ]];
    then
        echo "found ${folder_name}_1"
        found_dir=${found_dir}_1
        rename=1 #flag to rename the folder after
    else
        rename=0
        echo "no extra found"
    fi

    echo "cp -r '$found_dir' '${dicom_dir}/${subject}/'"
    cp -r "$found_dir" "${dicom_dir}/${subject}/"
    
    #rename:
    if [[ $rename -eq 1 ]]; then
        echo "mv ${dicom_dir}/${subject}/${folder_name}_1 ${dicom_dir}/${subject}/${folder_name}"
        mv "${dicom_dir}/${subject}/${folder_name}_1" "${dicom_dir}/${subject}/${folder_name}"
    fi
    done
done

# Actions for each existing scan session:
for folder_name in "${list_to_extract[@]}"; do
    echo "------------- ${folder_name} start ---------------"  
    # Use find command to search for folders matching the name
    find "${dicom_dir}/${subject}" -type d -name "$folder_name" 2>/dev/null | while read -r found_dicom_dir; do

    
    # 3. unzip all the dicom.zip (-q to remove verbose)
    echo "Unzipping dicom.zip for '$found_dicom_dir' "

    zip_file=$(find "${dicom_dir}/${subject}/${folder_name}" -name "*dicom.zip")
    echo "Unzipping dicom.zip for '$found_dicom_dir' "
    echo "zip file name is $zip_file"
    echo "command: unzip -q "$zip_file" -d "${dicom_dir}/${subject}/${folder_name}" "
    unzip -q "${zip_file}" -d "${dicom_dir}/${subject}/${folder_name}"
    echo "unzipping finished" 

    # 4. run dcm2niix on dicom files
    dcm_without_zip=$(find "${dicom_dir}/${subject}/${folder_name}" -name "*dicom")
    dcm_file_name=$(basename "$dcm_without_zip")
    echo "Converting $dcm_without_zip"
    dcm2niix "${dcm_without_zip}"
    

    # 5. move the nii and json files to BIDS_dir:
    # rename the nii file and mv to bids folder

    # find the nifti and json
    # take the first one on the list
    nii_file=$(find "$dcm_without_zip" -type f -name "*.nii" | head -n 1)
    json_file=$(find "$dcm_without_zip" -type f -name "*.json" | head -n 1)
   
    echo "nii file: "
    echo "$nii_file"
    echo "json file: "
    echo "$json_file"


    # mv nii and json to BIDS_dir
    case $folder_name in
        "ORIG 3D Ax T1 BRAVO")
            echo "------------- T1w BRAVO finished -------------"    
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_T1w.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_T1w.json

        ;;

        "ORIG 3D Ax T2 Cube")
            echo "------------- T2w CUBE finished -------------"   
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_T2w.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_T2w.json

        ;;

        "ORIG_ DWI")
            echo "------------- dwi finished -------------"   
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/dwi/sub-${subject}_ses-${session}_dwi.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/dwi/sub-${subject}_ses-${session}_dwi.json

            # copy bval and bvec data:  TODO - find the bval file directory in the sync folder
            # cp ${dwi_bval} ${BIDS_dir}/sub-${subject}/dwi/sub-${subject}_ses-${session}_dwi.bval
            # cp ${dwi_bvec} ${BIDS_dir}/sub-${subject}/dwi/sub-${subject}_ses-${session}_dwi.bvec

        ;;

        "ORIG_DWI PEpolar change CV")
            echo "------------- dwi pepolar -----------"
            dir=PA
        
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.json

            # delete "IntendedFor" line in json file 
            sed -i '/IntendedFor/d' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.json 
            # add back "IntendedFor" for all func files
            sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/dwi\/${subject}_ses-${session}_dwi.nii"'"],\n\t"SAR"/' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.json

        ;;
        "Pressure fMRI")
            echo "-------------Pressure fMRI finished -------------"   
            task=pressure; run=1;
            # mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.nii
            # mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.json
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_bold.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_bold.json
            # # copy Retroicor
            # if [[ -f ${Pfile}.ret.nii ]]; then
            #     echo "cp ${Pfile}.ret.nii ${output_dir}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_bold.nii"
            #     cp ${Pfile}.ret.nii ${output_dir}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_bold.nii
            # fi
            # if [[ -f ${Pfile}.den.nii ]]; then
            #     echo "cp ${Pfile}.den.nii ${output_dir}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_bold.nii"
            #     cp ${Pfile}.den.nii ${output_dir}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_bold.nii
            # fi
            
        ;;
        "Pressure fMRI PEpolar")
            echo "-------------- Pressure fMRI PEpolar finished -------------"   
            acq=pressure; dir=PA; run=1; task=pressure;
            
            # mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.nii
            # mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json
            
            # delete "IntendedFor" line in json file 
            # sed -i '/IntendedFor/d' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json 
            # # add back "IntendedFor" for all func files
            # sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"'"],\n\t"SAR"/' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.json
            sed -i '/IntendedFor/d' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.json 
            sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_bold.nii"'"],\n\t"SAR"/' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.json

        
        ;; 
        "Rest1")
            echo "-------------- Rest1 finished -------------"
            task=rest; run=1 
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.json

            # if [[ -f ${folder}/${Pfile}.ret.nii ]]; then
            #     echo "cp ${folder}/${Pfile}.ret.nii ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"
            #     cp ${folder}/${Pfile}.ret.nii ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii
            # fi
            # if [[ -f ${folder}/${Pfile}.den.nii ]]; then
            #     echo "cp ${folder}/${Pfile}.den.nii ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"
            #     cp ${folder}/${Pfile}.den.nii ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii
            # fi
            # # copy func json file from Rest1 for Retroicor & RetroRV
            # cp ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.json ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.json
            # cp ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.json ${output_dir}func/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.json


        ;; 
        "Rest1 PEpolar")
            echo "-------------- Rest1 PEpolar finished -------------"
            acq=rest; run=1; dir=PA; task=rest;
            
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json

            # delete "IntendedFor" line in json file 
            sed -i '/IntendedFor/d' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json 
            # add back "IntendedFor" for all func files
            sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"'"],\n\t"SAR"/' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json
            
        ;; 
        "Rest2")
            echo "-------------- Rest2 finished ---------------------"
            task=rest; run=2
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.json

            # if [[ -f ${Pfile}.ret.nii ]]; then
            #     echo "cp ${Pfile}.ret.nii ${output_dir}/${folder}/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"
            #     cp ${Pfile}.ret.nii ${output_dir}/${folder}/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii
            # fi
            # if [[ -f ${Pfile}.den.nii ]]; then
            #     echo "cp ${Pfile}.den.nii ${output_dir}/${folder}/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"
            #     cp ${Pfile}.den.nii ${output_dir}/${folder}/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii
            # fi
            # # copy func json file from Rest2 for Retroicor & RetroRV
            # cp ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.json ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.json
            # cp ${output_dir}/func/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.json ${output_dir}func/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.json

                    
        ;;
        
        "Rest2 PEpolar")
            echo "-------------- Rest2 PEpolar finished -------------"
            acq=rest; run=2; dir=PA; task=rest;
            mv "$nii_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.nii
            mv "$json_file" ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json

            # delete "IntendedFor" line in json file 
            sed -i '/IntendedFor/d' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json 
            # add back "IntendedFor" for all func files
            sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_run-${run}_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"'", "'"ses-${session}\/func\/sub-${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"'"],\n\t"SAR"/' ${BIDS_dir}/sub-${subject}/ses-${session}/fmap/sub-${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json
            
        ;;
        *)
            # Default action if no match is found
            echo "------------- $folder_name finished ------------"
            ;;

    esac
    done

    


done    


# create participant and dataset json and tsv
python BIDS_annotation.py ${subject} ${BIDS_dir}


# run BIDS validator through the python script
BIDS_validation=$(python run_BIDS_validator.py $BIDS_dir)
echo "BIDS validated: $BIDS_validation"


# remove the unzipped dicom
echo "removing dicom folder: rm -r ${dicom_dir}/${subject}/ "
rm -r "${dicom_dir}/${subject}/"




