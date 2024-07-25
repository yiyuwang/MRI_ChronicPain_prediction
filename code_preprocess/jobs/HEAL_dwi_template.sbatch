#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=${subject}_dwi_eddycorrect
#################  
#a file for job output, you can check job progress
#SBATCH --output=output/${subject}_dwi_eddycorrect_%A.out 
#################
# a file for errors from the job
#SBATCH --error=output/${subject}_dwi_eddycorrect_%A.err
#################
#time: hh:mm:ss
#SBATCH --time=24:00:00
#################
#quality of service
#SBATCH --qos=normal
#################
#SBATCH --mem=64000
#################
start=`date +%s`



# Yiyu Wang 2023-09-11


#module --ignore_cache load "singularity"
module load system
unset PYTHONPATH

module load contribs poldrack
module load fsl/6.0.6.2

mkdir $output_dir

echo $output_dir

echo 'running eddycorrect'
eddy_correct ${input_dir}/${subject}_ses-01_dwi.nii ${output_dir}/${subject} 0

# calculating fd, rotation, translation:
# modified script from here: https://github.com/Woutervdbos/DTI-Motion/tree/master


sh /scratch/groups/smackey/HEAL/code/jobs/HEAL_ec_plot.sh ${output_dir}/${subject}.ecclog ${output_dir}

end=`date +%s`

