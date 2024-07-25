#!/usr/bin/bash
#SBATCH --job-name=HEAL_pressure
#SBATCH --output=output/HEAL_pressure_%A.out 
#SBATCH --error=output/HEAL_pressure_%A.err
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

start=`date +%s`

#Yiyu Wang 2024-02-06

source activate /scratch/users/yiyuw/MyCondaEnvs/prediction_env
cd /scratch/groups/smackey/HEAL/code/jobs

python pressure_1stlvl_univariate.py

