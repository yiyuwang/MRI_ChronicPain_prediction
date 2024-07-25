#!/usr/bin/bash
#SBATCH --job-name=HEAL_DTW
#SBATCH --output=output/HEAL_DTW_%A.out 
#SBATCH --error=output/HEAL_DTW_%A.err
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

start=`date +%s`

#Yiyu Wang 2024-01-12

source activate /scratch/users/yiyuw/MyCondaEnvs/prediction_env
cd /scratch/groups/smackey/HEAL/code/jobs

python CreateDTW.py 'schaefer100'


