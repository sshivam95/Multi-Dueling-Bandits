#!/bin/sh
# SBATCH -J "Run Multi-Dueling Bandits experiments"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=16G
#SBATCH -A hpc-prf-pbrac
#SBATCH -p normal
#SBATCH -t 0-02:30:00
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

module reset
module load lang/Python/3.10.4-GCCcore-11.3.0-bare
cd ..
source venv/bin/activate
cd $PFS_FOLDER/Multi-Dueling-Bandits/

export ALGORITHM=$1
export SOLVER=$2
export SUBSET_SIZE=$3
export ARMS=$4

module list
python3 test_cluster.py -a $ALGORITHM --dataset $SOLVER --subset-size $SUBSET_SIZE --reps 50 --arms $ARMS
deactivate
exit 0
~