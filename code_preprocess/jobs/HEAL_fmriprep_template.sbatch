#!/usr/bin/bash
#SBATCH --job-name=${subject}_fMRIprep
#SBATCH --output=output/${subject}_fMRIprep_fw_%A.out 
#SBATCH --error=output/${subject}_fMRIprep_fw_%A.err
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

start=`date +%s`

#Yiyu Wang 2023-09-11

#module --ignore-cache load singularity
module load system
unset PYTHONPATH

mkdir ${work_dir}/sub-${subject}

singularity run \
--bind $project_dir \
--cleanenv ${project_dir}/sing_imgs/fmriprep-23.1.2.simg \
$BIDS_dir $output_dir \
participant \
--participant-label sub-${subject} \
-w ${work_dir}/sub-${subject} \
--output-spaces T1w MNI152NLin6Asym:res-2 MNI152NLin2009cAsym fsaverage \
--skip-bids-validation \
--cifti-output 91k \
--fs-license-file ${project_dir}/license.txt \
--fd-spike-threshold 0.5 --dvars-spike-threshold 2 \
--nthread 16 \
--mem_mb 64000 \
-vvv \
--stop-on-first-crash 

#--debug all \


end=`date +%s`