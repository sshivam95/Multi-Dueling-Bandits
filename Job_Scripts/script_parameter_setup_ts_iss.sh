declare -a algorithm_classes=("ThompsonSampling" "ThompsonSamplingContextual" "IndependentSelfSparring" "IndependentSelfSparringContextual")
declare -a solver_array=("saps" "mips")
declare -a subset_size=("2" "3" "4" "5" "6" "7" "8" "9" "10" "16")

for algorithm in "${algorithm_classes[@]}";
do
  for solver in "${solver_array[@]}";
  do
    for size in "${subset_size[@]}";
    do
      sbatch run_script_ts_iss.sh $algorithm $solver $size
    done
  done
done
