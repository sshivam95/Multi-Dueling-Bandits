#!/bin/sh
# SBATCH -J "Run Multi-Dueling Bandits experiments"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH -A hpc-prf-pbrac
#SBATCH -p normal
#SBATCH -t 0-00:00:25
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

cd $PFS_FOLDER/Multi-Dueling-Bandits/
module reset
module load lang/Python/3.10.4-GCCcore-11.3.0-bare
# export IMG_FILE=$PFS_FOLDER/CPPL/singularity/pbcppl.sif
# export SCRIPT_FILE=$PFS_FOLDER/CPPL/test.py
source venv/bin/activate

export ALGORITHM=$1
export SOLVER=$2
export SUBSET_SIZE=$3

module list
python3 test_cluster.py -a $ALGORITHM --dataset $SOLVER --subset-size $SUBSET_SIZE --reps 50
# singularity exec -B $PFS_FOLDER/CPPL/ --nv $IMG_FILE pipenv run python3 $SCRIPT_FILE -a=$ALGORITHM --dataset=$SOLVER --subset-size=$SUBSET_SIZE
deactivate
exit 0
~