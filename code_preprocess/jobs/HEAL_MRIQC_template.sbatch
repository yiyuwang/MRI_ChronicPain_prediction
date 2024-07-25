#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=${subject}_MRIQC
#################  
#a file for job output, you can check job progress
#SBATCH --output=output/${subject}_MRIQC.out 
#################
# a file for errors from the job
#SBATCH --error=output/${subject}_MRIQC.err
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


singularity run /scratch/groups/smackey/HEAL/sing_imgs/mriqc-22.0.6.simg $BIDS_dir $MRIQC_dir participant --participant_label $subject -w $work_dir --mem_gb 64 --nprocs 8 --ants-nthreads 8 --fd_thres 0.5 -v

end=`date +%s`
