import argparse
import inspect
import logging
import sys
from time import perf_counter
from typing import Generator

import numpy as np
from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm import tqdm

from algorithms import regret_minimizing_algorithms
from util import utility_functions
from util.constants import JointFeatureMode, Solver
import pandas as pd

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s | %(name)s (%(levelname)s):\t %(message)s"
)


def _main():

    parser = argparse.ArgumentParser(
        description="Run Multi-Dueling Bandits experiments."
    )
    algorithm_names_to_algorithms = {
        algorithm.__name__: algorithm for algorithm in regret_minimizing_algorithms
    }
    algorithm_choices_string = " ".join(algorithm_names_to_algorithms.keys())
    solver_list = [solver.value for solver in Solver]
    subset_size_list = [5, 6, 7, 8, 9, 10, 16]
    parser.add_argument(
        "-a, --algorithms",
        nargs="+",
        metavar="ClassName",
        dest="algorithms",
        default=algorithm_names_to_algorithms.keys(),
        help=f"Algorithms to compare. (default: {algorithm_choices_string})",
        choices=algorithm_names_to_algorithms.keys(),
    )
    parser.add_argument(
        "--reps",
        dest="reps",
        default=10,
        type=int,
        help="How often to run each algorithm. Results will be averaged. (default: 10)",
    )
    parser.add_argument(
        "--subset-size",
        dest="subset_size",
        default=subset_size_list,
        type=int,
        help=f"How many arms the generated subset from original arms should contain. (default: {subset_size_list})",
    )
    parser.add_argument(
        "--dataset",
        dest="solver",
        nargs="+",
        default=solver_list,
        type=str,
        choices=solver_list,
        help=f"Which dataset of problem instances to use for experimentation. (default: {solver_list})",
    )
    parser.add_argument(
        "-jfm, --joint-feature-mode",
        dest="joint_featured_map_mode",
        default=JointFeatureMode.POLYNOMIAL.value,
        type=str,
        help="Which joint feature map mode to use for mapping context features. (default: polynomial)",
    )
    parser.add_argument(
        "--random-seed",
        dest="base_random_seed",
        default=42,
        type=int,
        help="Base random seed for reproducible results.",
    )
    parser.add_argument(
        "--jobs",
        dest="n_jobs",
        default=-1,
        type=int,
        help="How many experiments to run in parallel. The special value -1 stands for the number of processor cores.",
    )

    args = parser.parse_args()
    algorithms = [
        algorithm_names_to_algorithms[algorithm] for algorithm in args.algorithms
    ]
 
    # if not isinstance(solver, list):
    #     solver_list = list()
    #     solver_list.append(solver for solver in solver.keys())
    # else:
    solver_list = args.solver
    if not isinstance(args.subset_size, list):
        subset_size_list = [args.subset_size]
    else:
        subset_size_list = args.subset_size
    run_experiment(
        algorithms=algorithms,
        reps=args.reps,
        joint_featured_map_mode=args.joint_featured_map_mode,
        base_random_seed=args.base_random_seed,
        n_jobs=args.n_jobs,
        solver_list=solver_list,
        subset_size_list=subset_size_list
    )


def run_experiment(
    algorithms,
    reps,
    joint_featured_map_mode,
    base_random_seed,
    n_jobs,
    solver_list,
    subset_size_list
):
    start_time = perf_counter()
    regrets = np.empty(reps, dtype=np.ndarray)
    execution_times = np.zeros(reps)
    
    def job_producer() -> Generator:
        for solver in solver_list:
            if solver == Solver.SAPS.value:
                parametrizations = utility_functions.get_parameterization_saps()
                features = utility_functions.get_features_saps()
                running_time = utility_functions.get_run_times_saps()
            else:
                parametrizations = utility_functions.get_parameterization_mips()
                features = utility_functions.get_features_mips()
                running_time = utility_functions.get_run_times_mips()
            for algorithm_class in algorithms:
                algorithm_name = algorithm_class.__name__
                for subset_size in subset_size_list:#[5, 6, 7, 8, 9, 10, 16]:
                    for rep_id in range(reps):
                        random_state = np.random.RandomState(
                            (base_random_seed + hash(algorithm_name) + rep_id) % 2**32
                        )
                        parameters = {
                            "joint_featured_map_mode": joint_featured_map_mode,
                            "solver": solver,
                            "subset_size": subset_size,
                            "parametrizations": parametrizations,
                            "features": features,
                            "running_time": running_time,
                        }
                        yield delayed(single_experiment)(
                            random_state,
                            algorithm_class,
                            algorithm_name,
                            parameters,
                            rep_id
                        )

    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()

    jobs = list(job_producer())
    with tqdm_joblib(tqdm(desc="Run Multi-Dueling Bandits experiments", total=len(jobs))) as progress_bar:
        result = Parallel(n_jobs=n_jobs, verbose=50)(jobs)
    runtime = perf_counter() - start_time
    result_df = pd.concat(result)
    algorithm_name = result_df["algorithm"].unique()
    solver_name = result_df["solver"].unique()
    subset_size_list = result_df["subset_size"].unique()
    print("Saving files...")
    for solver in solver_name:
        for name in algorithm_name:
            for subset_size in subset_size_list:
                for rep_id in range(reps):
                    mask = (result_df["solver"] == solver) & (result_df["algorithm"] == name) & (result_df["subset_size"] == subset_size) & (result_df["rep_id"] == rep_id)
                    regrets[rep_id] = result_df[mask]["regret"].to_numpy()
                    execution_times[rep_id] = result_df[mask]["execution_time"].mean()
            np.save(f"Regret_results_theta0//regret_{name}_{solver}_{subset_size}", regrets)
            np.save(
                f"Execution_times_results_theta0//execution_time_{name}_{solver}_{subset_size}",
                execution_times,
            )
    print(f"Experiments took {round(runtime)}s.")


def single_experiment(
    random_state,
    algorithm_class,
    algorithm_name,
    parameters,
    rep_id
):
    solver = parameters["solver"]
    subset_size = parameters["subset_size"]
    print(f"{algorithm_name} with {solver} and {subset_size} started...")
    
    parameters["random_state"] = random_state
    parameters_to_pass = dict()
    for parameter in parameters.keys():
        if parameter in inspect.getfullargspec(algorithm_class.__init__)[0]:
            parameters_to_pass[parameter] = parameters[parameter]
    algorithm_object = algorithm_class(**parameters_to_pass)
    algorithm_object.run()
    regret = algorithm_object.get_regret()
    execution_time = algorithm_object.execution_time
    print(f"Rep {rep_id}: {algorithm_name} with {solver} and {subset_size} finished...")
    data_frame = pd.DataFrame(
        {
            "rep_id": rep_id,
            "algorithm": algorithm_name,
            "solver": solver,
            "subset_size": subset_size,
            "regret": regret,
            "execution_time": execution_time,
        }
    )
    return data_frame


if __name__ == "__main__":
    _main()
