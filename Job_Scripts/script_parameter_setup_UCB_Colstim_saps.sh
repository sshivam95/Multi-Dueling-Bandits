declare -a algorithm_classes=("UCB" "Colstim")
declare -a solver_array=("saps")
declare -a subset_size=("2" "3" "4" "5" "6" "7" "8" "9" "10" "16")

for algorithm in "${algorithm_classes[@]}";
do
  for solver in "${solver_array[@]}";
  do
    for size in "${subset_size[@]}";
    do
      sbatch run_script_UCB_Colstim_saps.sh $algorithm $solver $size
    done
  done
done
