#!/usr/bin/bash
#SBATCH --job-name=HEAL_parcellation
#SBATCH --output=output/HEAL_parcellation_%A.out 
#SBATCH --error=output/HEAL_parcellation_%A.err
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

start=`date +%s`

#Yiyu Wang 2024-01-12

source activate /scratch/users/yiyuw/MyCondaEnvs/prediction_env
cd /scratch/groups/smackey/HEAL/code/jobs

python CreateParcellationTimeSeries.py 'rest' 'schaefer' 100


