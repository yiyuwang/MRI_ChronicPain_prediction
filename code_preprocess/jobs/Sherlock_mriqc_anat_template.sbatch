#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=${subject}_MRIqc_T1w
#################  
#a file for job output, you can check job progress
#SBATCH --output=${subject}_MRIqc_T1w.out 
#################
# a file for errors from the job
#SBATCH --error=${subject}_MRIqc_T1w.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=48:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#SBATCH --mem=64000
#################
start=`date +%s`
#module --ignore-cache load singularity
module load system
unset PYTHONPATH


singularity run /home/groups/russpold/singularity_images/mriqc-22.0.6.simg $bids_anat $output_anat participant --participant_label $subject -w $work_anat --mem_gb 32 -m T1w --nprocs 8 --ants-nthreads 8 -v   

end=`date +%s`
