declare -a num_arms=("15" "20")
declare -a algorithm_classes=("Colstim_v2")
declare -a solver_array=("saps")
declare -a subset_size=("2" "3" "4" "5" "6" "7" "8" "9" "10")

for arms in "${num_arms[@]}";
do
  for algorithm in "${algorithm_classes[@]}";
  do
    for solver in "${solver_array[@]}";
    do
      for size in "${subset_size[@]}";
      do
        sbatch run_script_Colstim_v2_saps.sh $algorithm $solver $size $arms
      done
    done
  done
done