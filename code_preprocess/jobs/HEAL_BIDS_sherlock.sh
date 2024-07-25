#!/bin/bash

# Yiyu Wang 7/1/2024
# run BIDS conversion on pfile on sherlock


function usage()
{
cat << EOF

USAGE
  `basename ${0}`

  prepare data for running fmriprep 
  # steps to set up sherlock:
  
  * important: assume Exam directory (Exxxxx) is already uploaded to sherlock:/scratch/groups/smackey/HEAL_MRI_BIDS
               run in directory sherlock /scratch/groups/smackey/HEAL_MRI_BIDS/
               scp -r Z:\Mackeylab\Scans\3T2\Exxxxx XXXX@sherlock.stanford.edu:/scratch/groups/smackey/HEAL_MRI_BIDS/


  steps:
  1. Open windows powershell and run this:
  scp -r Z:\Mackeylab\Scans\3T2\Exxxxx XXXX@sherlock.stanford.edu:/scratch/groups/smackey/HEAL_MRI_BIDS/
  (keep the powershell open until all files are uploaded)
  (recommand run this for a couple of subjects at the same time (in multiple powershell windows) to save time)

  2. Go to CSF.pdf, and complete the command line based on the SUMMARY file:
  " sh HEAL_BIDS_sherlock.sh -s sub-bio0268 -x 01 -E E14225 -B 7 -C 14 -D 8 -d 9 "
  " sh HEAL_BIDS_sherlock.sh -s sub-bio0274 -x 01 -E E14263 -B 6 -C 13 -D 7 -d 8 "
 
  3. Go to sherlock terminal on the ondemand:
  cd /scratch/groups/smackey/HEAL_MRI_BIDS
  sh_dev
  module load biology
  module load dcm2niix
  sh HEAL_BIDS_sherlock.sh -s sub-0003 -x 01 -E E10678 -B 7 -C 12 -D 8 -d 10
  (this should be very quick)

  4. Download the BIDS back to pain server
  # after completion on your local powershell: run this command (run this after a batch of subjects have been processed)
  rsync -a --progress XXXX@sherlock.stanford.edu:/scratch/groups/smackey/HEAL_MRI_BIDS/fmriprep Z:\Mackeylab\PROJECTS\HEAL_data\MRI\BIDS_luis


MANDATORY ARGUMENTS
  -s <subject>			    Subject Study ID (sub-####)
  -x <session>			    Session (e.g. 01)
  -E <exam_num>         Exam number (Exxxxx)

OPTIONAL ARGUMENTS

  -B <T1>               T1 Bravo Series number (e.g. 7). Skip if empty.
  -C <T2>               T2 Cube Series number (e.g. 12). Skip if empty.
  -D <dwi>              DWI series number (e.g. 8). Skip if empty.
  -d <dwi_pepolar>      DWI pepolar series number (e.g. 10). Skip if empty.
  
EOF
}

if [ ! ${#@} -gt 0 ]; then
    usage `basename ${0}`
    exit 1
fi

#Initialization of variables

scriptname=${0}
subject=
session=
exam_num=
T1=
T2=
dwi=
dwi_pepolar=

while getopts “hs::x::E::B::C::D::d::” OPTION
do
	case $OPTION in
	 h)
			usage
			exit 1
			;;
	 s)
		subject=$OPTARG
			;;
	 x)
		session=$OPTARG
			;;
   E)
		exam_num=$OPTARG
			;;   
   B)
		T1=$OPTARG
			;;   
   C)
		T2=$OPTARG
			;;   
   D)
		dwi=$OPTARG
			;;   
   d)
		dwi_pepolar=$OPTARG
			;;   
	 ?)
		 usage
		 exit
		 ;;
     esac
done

# Set default values
if [[ -z ${T1} ]]; then
    T1=0                       
fi
echo T1=${T1}
if [[ -z ${T2} ]]; then
    T2=0                       
fi
echo T2=${T2}
if [[ -z ${dwi} ]]; then
    dwi=0                       
fi
echo dwi=${dwi}
if [[ -z ${dwi_pepolar} ]]; then
    dwi_pepolar=0                       
fi
echo dwi_pepolar=${dwi_pepolar}

# # Check the parameters

if [[ -z ${subject} ]]; then
     echo "ERROR: Subject not specified. Exit program."
     exit 1
fi
if [[ -z ${session} ]]; then
    echo "ERROR: Session not specified. Exit program."
    exit 1
fi
if [[ -z ${exam_num} ]]; then
	 echo "ERROR: Exam_num not specified. Exit program."
     exit 1
fi



if [[ ! -d fmriprep/${subject} ]]; then 
  echo "create fmriprep/${subject}"
  mkdir -p fmriprep/${subject}/ses-${session}
  mkdir -p fmriprep/${subject}/ses-${session}/anat
  mkdir -p fmriprep/${subject}/ses-${session}/func
  mkdir -p fmriprep/${subject}/ses-${session}/fmap
  mkdir -p fmriprep/${subject}/ses-${session}/dwi
fi


T1_dir=`printf %03d $T1`
T2_dir=`printf %03d $T2`
dwi_dir=`printf %03d $dwi`
dwi_pepolar_dir=`printf %03d $dwi_pepolar`

if [[ ${T1} -ne 0 ]]
then
  echo "------------- T1w BRAVO -----------"
  # remove old nii and json files if exist
  if [ `ls -1 ${exam_num}/anat/${T1_dir}/*nii 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${T1_dir}/*nii; fi
  if [ `ls -1 ${exam_num}/anat/${T1_dir}/*json 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${T1_dir}/*json; fi

  dcm2niix ${exam_num}/anat/${T1_dir}
  T1_anat=`ls ${exam_num}/anat/${T1_dir}/*.nii`
  cp ${T1_anat} fmriprep/${subject}/ses-${session}/anat/${subject}_ses-${session}_T1w.nii
  T1_json=`ls ${exam_num}/anat/${T1_dir}/*.json`
  cp ${T1_json} fmriprep/${subject}/ses-${session}/anat/${subject}_ses-${session}_T1w.json
fi

if [[ ${T2} -ne 0 ]]
then
  echo "------------- T2w CUBE -----------"
  # remove old nii and json files if exist
  if [ `ls -1 ${exam_num}/anat/${T2_dir}/*nii 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${T2_dir}/*nii; fi
  if [ `ls -1 ${exam_num}/anat/${T2_dir}/*json 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${T2_dir}/*json; fi

  dcm2niix ${exam_num}/anat/${T2_dir}
  T2_anat=`ls ${exam_num}/anat/${T2_dir}/*.nii`
  cp ${T2_anat} fmriprep/${subject}/ses-${session}/anat/${subject}_ses-${session}_T2w.nii
  T2_json=`ls ${exam_num}/anat/${T2_dir}/*.json`
  cp ${T2_json} fmriprep/${subject}/ses-${session}/anat/${subject}_ses-${session}_T2w.json
fi

if [[ ${dwi} -ne 0 ]]
then
  echo "------------- dwi -----------"
  # remove old nii and json files if exist
  if [ `ls -1 ${exam_num}/anat/${dwi_dir}/*nii 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${dwi_dir}/*nii; fi
  if [ `ls -1 ${exam_num}/anat/${dwi_dir}/*json 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${dwi_dir}/*json; fi
  if [ `ls -1 ${exam_num}/anat/${dwi_dir}/*bval 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${dwi_dir}/*bval; fi
  if [ `ls -1 ${exam_num}/anat/${dwi_dir}/*bvec 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${dwi_dir}/*bvec; fi

  dcm2niix ${exam_num}/anat/${dwi_dir}
  dwi_nii=`ls ${exam_num}/anat/${dwi_dir}/*.nii`
  cp ${dwi_nii} fmriprep/${subject}/ses-${session}/dwi/${subject}_ses-${session}_dwi.nii
  dwi_json=`ls ${exam_num}/anat/${dwi_dir}/*.json`
  cp ${dwi_json} fmriprep/${subject}/ses-${session}/dwi/${subject}_ses-${session}_dwi.json
  dwi_bval=`ls ${exam_num}/anat/${dwi_dir}/*.bval`
  cp ${dwi_bval} fmriprep/${subject}/ses-${session}/dwi/${subject}_ses-${session}_dwi.bval
  dwi_bvec=`ls ${exam_num}/anat/${dwi_dir}/*.bvec`
  cp ${dwi_bvec} fmriprep/${subject}/ses-${session}/dwi/${subject}_ses-${session}_dwi.bvec
fi

if [[ ${dwi_pepolar} -ne 0 ]]
then
  echo "------------- dwi pepolar -----------"
  dir=PA
  # remove old nii and json files if exist
  if [ `ls -1 ${exam_num}/anat/${dwi_pepolar_dir}/*nii 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${dwi_pepolar_dir}/*nii; fi
  if [ `ls -1 ${exam_num}/anat/${dwi_pepolar_dir}/*json 2>/dev/null | wc -l ` -gt 0 ]; then rm ${exam_num}/anat/${dwi_pepolar_dir}/*json; fi
  #if [ `ls -1 anat/${dwi_pepolar_dir}/*bval 2>/dev/null | wc -l ` -gt 0 ]; then rm anat/${dwi_pepolar_dir}/*bval; fi
  #if [ `ls -1 anat/${dwi_pepolar_dir}/*bvec 2>/dev/null | wc -l ` -gt 0 ]; then rm anat/${dwi_pepolar_dir}/*bvec; fi

  dcm2niix ${exam_num}/anat/${dwi_pepolar_dir}
  dwi_nii=`ls ${exam_num}/anat/${dwi_pepolar_dir}/*.nii`
  cp ${dwi_nii} fmriprep/${subject}/ses-${session}/fmap/${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.nii
  dwi_json=`ls ${exam_num}/anat/${dwi_pepolar_dir}/*.json`
  cp ${dwi_json} fmriprep/${subject}/ses-${session}/fmap/${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.json

  # delete "IntendedFor" line in json file 
  sed -i '/IntendedFor/d' fmriprep/${subject}/ses-${session}/fmap/${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.json 
  # add back "IntendedFor" for all func files
  sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/dwi\/${subject}_ses-${session}_dwi.nii"'"],\n\t"SAR"/' fmriprep/${subject}/ses-${session}/fmap/${subject}_ses-${session}_acq-dwi_dir-${dir}_epi.json
fi

echo "--------------- functional scans ------------------"
cd ${exam_num}
E_file=`ls E*`
#E_file=( E10632S013P43008.7 ) #( E10632S005P36864.7 )
echo ${E_file}



for file in ${E_file}
do
  echo "*********** ${file} *************"
  Series=`grep 'series description = ' ${file}`
  echo ${Series}
  
  Series_num=`echo $file | cut -f 2 -d 'S' | cut -f 1 -d 'P'`
  echo "Series_num = $Series_num"

  Series_name=`grep 'series description = ' ${file} | cut -f2 -d'='`
  # remove first empty character
  Series_name="${Series_name:1}"
  echo "Series_name = ${Series_name}"

  Pfile=P`echo ${file} | cut -f2 -d'P'`
  echo "Pfile = ${Pfile}"  
  echo "*******************************************"
  
  if [[ ${Series_name} == "Rest1" ]]; then 
    echo "-------------- Rest1 ---------------------"
    Rest1_Series_num=$Series_num
    echo "Rest1_Series_num = $Rest1_Series_num"
    Rest1_Series_num_no_zeros=$(echo $Rest1_Series_num | sed 's/^0*//')
    folder=func; task=rest; run=1 
    echo "cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_run-${run}_bold.nii"
    cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_run-${run}_bold.nii
    if [[ -f ${Pfile}.ret.nii ]]; then
     echo "cp ${Pfile}.ret.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"
      cp ${Pfile}.ret.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii
    fi
    if [[ -f ${Pfile}.den.nii ]]; then
     echo "cp ${Pfile}.den.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"
      cp ${Pfile}.den.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii
    fi
  fi

  
  if [[ ${Series_name} == "Rest2" ]]; then 
    echo "-------------- Rest2 ---------------------"
    Rest2_Series_num=$Series_num
    echo "Rest2_Series_num = $Rest2_Series_num"
    Rest2_Series_num_no_zeros=$(echo $Rest2_Series_num | sed 's/^0*//')
    folder=func; task=rest; run=2
    echo "cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_run-${run}_bold.nii"
    cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_run-${run}_bold.nii
    if [[ -f ${Pfile}.ret.nii ]]; then
     echo "cp ${Pfile}.ret.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"
      cp ${Pfile}.ret.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii
    fi
    if [[ -f ${Pfile}.den.nii ]]; then
     echo "cp ${Pfile}.den.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"
      cp ${Pfile}.den.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii
    fi
  fi

  
  if [[ ${Series_name} == "Rest1 PEpolar" ]]; then 
    echo "-------------- Rest1 PEpolar -------------"
    Rest1_PEpolar_Series_num=$Series_num
    echo "Rest1_PEpolar_Series_num = $Rest1_PEpolar_Series_num"
    Rest1_PEpolar_Series_num_no_zeros=$(echo $Rest1_PEpolar_Series_num | sed 's/^0*//')
    folder=fmap; acq=rest; run=1; dir=PA; 
    echo "cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.nii"
    cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.nii
    # remove old nii and json files
    find anat/${Series_num} -name "*.nii" -exec rm -rf {} \;
    find anat/${Series_num} -name "*.json" -exec rm -rf {} \;
    # run dcm2niix to get json file
    echo "run dcm2niix on Series ${Series_num}"
    dcm2niix anat/${Series_num}
    # get json file name
    json_name=`echo anat/${Series_num}/*.json | cut -f3 -d '/'`
    # copy json file
    cp anat/${Series_num}/${json_name} ../fmriprep/${subject}/ses-${session}/${folder}/
    mv ../fmriprep/${subject}/ses-${session}/${folder}/${json_name} ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json
    # delete "IntendedFor" line in json file 
    sed -i '/IntendedFor/d' ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json 
    # add back "IntendedFor" for all func files
    sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_run-${run}_bold.nii"'", "'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"'", "'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"'"],\n\t"SAR"/' ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json

    # copy json file for Rest1
    cp ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    # modify parameters for Rest1 json file
    sed -i '/IntendedFor/d' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i 's/"Rest1_PEpolar"/"Rest1"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i 's/"Flipped"/"Unflipped"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i 's/"j"/"j-"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i "s/: ${Rest1_PEpolar_Series_num_no_zeros}/: ${Rest1_Series_num_no_zeros}/g" ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    # add TaskName
    if grep TaskName ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json; then echo TaskName exists; else sed -i 's/"SAR"/"TaskName": "'"${task}"'",\n\t"SAR"/' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json; fi

  fi

  
  if [[ ${Series_name} == "Rest2 PEpolar" ]]; then 
    echo "-------------- Rest2 PEpolar -------------"
    Rest2_PEpolar_Series_num=$Series_num
    echo "Rest2_PEpolar_Series_num = $Rest2_PEpolar_Series_num"
    Rest2_PEpolar_Series_num_no_zeros=$(echo $Rest2_PEpolar_Series_num | sed 's/^0*//')
    folder=fmap; acq=rest; run=2; dir=PA; 
    echo "cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.nii"
    cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.nii
    # remove old nii and json files
    find anat/${Series_num} -name "*.nii" -exec rm -rf {} \;
    find anat/${Series_num} -name "*.json" -exec rm -rf {} \;    
    #run dcm2niix to get json file
    echo "run dcm2niix on Series ${Series_num}"    
    dcm2niix anat/${Series_num}
    # get json file name
    json_name=`echo anat/${Series_num}/*.json | cut -f3 -d '/'`
    # copy json file
    cp anat/${Series_num}/${json_name} ../fmriprep/${subject}/ses-${session}/${folder}/
    mv ../fmriprep/${subject}/ses-${session}/${folder}/${json_name} ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json
    # delete "IntendedFor" line in json file 
    sed -i '/IntendedFor/d' ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json 
    # add back "IntendedFor" for all func files
    sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_run-${run}_bold.nii"'", "'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_rec-Retroicor_run-${run}_bold.nii"'", "'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_rec-RetroRV_run-${run}_bold.nii"'"],\n\t"SAR"/' ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json
  
    # copy json file for Rest2
    cp ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_run-${run}_epi.json ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    # modify parameters for Rest2 json file
    sed -i '/IntendedFor/d' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i 's/"Rest2_PEpolar"/"Rest2"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i 's/"Flipped"/"Unflipped"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i 's/"j"/"j-"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    sed -i "s/: ${Rest2_PEpolar_Series_num_no_zeros}/: ${Rest2_Series_num_no_zeros}/g" ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json
    # add TaskName
    if grep TaskName ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json; then echo TaskName exists; else sed -i 's/"SAR"/"TaskName": "'"${task}"'",\n\t"SAR"/' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_run-${run}_bold.json; fi

  fi

  
  if [[ ${Series_name} == "Pressure fMRI" ]]; then 
    echo "-------------Pressure fMRI -------------"
    Pressure_Series_num=$Series_num
    echo "Pressure_Series_num = $Pressure_Series_num"
    Pressure_Series_num_no_zeros=$(echo $Pressure_Series_num | sed 's/^0*//')
    folder=func; task=pressure
    echo "cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_bold.nii"
    cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_bold.nii
    if [[ -f ${Pfile}.ret.nii ]]; then
     echo "cp ${Pfile}.ret.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_bold.nii"
      cp ${Pfile}.ret.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-Retroicor_bold.nii
    fi
    if [[ -f ${Pfile}.den.nii ]]; then
     echo "cp ${Pfile}.den.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_bold.nii"
      cp ${Pfile}.den.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_task-${task}_rec-RetroRV_bold.nii
    fi
  fi

  if [[ ${Series_name} == "Pressure fMRI PEpolar" ]]; then 
    echo "-------------- Pressure fMRI PEpolar -------------"
    Pressure_PEpolar_Series_num=$Series_num
    echo "Pressure_PEpolar_Series_num = $Pressure_PEpolar_Series_num"
    Pressure_PEpolar_Series_num_no_zeros=$(echo $Pressure_PEpolar_Series_num | sed 's/^0*//')
    folder=fmap; acq=pressure; dir=PA; 
    echo "cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.nii"
    cp ${Pfile}.nii ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.nii
    # remove old nii and json files
    find anat/${Series_num} -name "*.nii" -exec rm -rf {} \;
    find anat/${Series_num} -name "*.json" -exec rm -rf {} \;    
    #run dcm2niix to get json file
    echo "run dcm2niix on Series ${Series_num}"    
    dcm2niix anat/${Series_num}
    # get json file name
    json_name=`echo anat/${Series_num}/*.json | cut -f3 -d '/'`
    # copy json file
    cp anat/${Series_num}/${json_name} ../fmriprep/${subject}/ses-${session}/${folder}/
    mv ../fmriprep/${subject}/ses-${session}/${folder}/${json_name} ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.json
    # delete "IntendedFor" line in json file 
    sed -i '/IntendedFor/d' ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.json 
    # add back "IntendedFor" for all func files
    sed -i 's/"SAR"/"IntendedFor": ["'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_bold.nii"'", "'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_rec-Retroicor_bold.nii"'", "'"ses-${session}\/func\/${subject}_ses-${session}_task-${task}_rec-RetroRV_bold.nii"'"],\n\t"SAR"/' ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.json
  
    # copy json file for Pressure fMRI
    cp ../fmriprep/${subject}/ses-${session}/${folder}/${subject}_ses-${session}_acq-${acq}_dir-${dir}_epi.json ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json
    # modify parameters for Pressure fMRI json file
    sed -i '/IntendedFor/d' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json
    sed -i 's/"Pressure fMRI PEpolar"/"Pressure fMRI"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json
    sed -i 's/"Flipped"/"Unflipped"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json
    sed -i 's/"j"/"j-"/g' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json
    sed -i "s/: ${Pressure_PEpolar_Series_num_no_zeros}/: ${Pressure_Series_num_no_zeros}/g" ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json
    # add TaskName
    if grep TaskName ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json; then echo TaskName exists; else sed -i 's/"SAR"/"TaskName": "'"${task}"'",\n\t"SAR"/' ../fmriprep/${subject}/ses-${session}/func/${subject}_ses-${session}_task-${task}_bold.json; fi

  fi

done

### Now log onto sherlock.stanford.edu
### cd /scratch/groups/smackey/P01/ScanData/fmriprep_cl
### cd ${subject}
### sbatch Sherlock_fmriprep_HN_${subject}.sbatch
