import argparse
import contextlib
import inspect
import logging
import sys
from time import perf_counter
from typing import Generator

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from algorithms import algorithm, regret_minimizing_algorithms
from util import utility_functions
from util.constants import JointFeatureMode, Solver

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
        type=int,
        default=10,
        help=f"How many arms the generated subset from original arms should contain. (default: 10)",
    )
    parser.add_argument(
        "--dataset",
        dest="solver",
        default=Solver.SAPS.value,
        type=str,
        help=f"Which dataset of problem instances to use for experimentation. (default: {Solver.SAPS.value})",
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

    solver = args.solver
    subset_size = args.subset_size
    run_experiment(
        algorithms=algorithms,
        reps=args.reps,
        n_jobs=args.n_jobs,
        joint_featured_map_mode=args.joint_featured_map_mode,
        base_random_seed=args.base_random_seed,
        solver=solver,
        subset_size=subset_size,
    )


def run_experiment(
    algorithms,
    reps,
    n_jobs,
    joint_featured_map_mode,
    base_random_seed,
    solver,
    subset_size,
):
    start_time = perf_counter()
    if solver == Solver.SAPS.value:
        parametrizations = utility_functions.get_parameterization_saps()
        features = utility_functions.get_features_saps()
        running_time = utility_functions.get_run_times_saps()
    else:
        parametrizations = utility_functions.get_parameterization_mips()
        features = utility_functions.get_features_mips()
        running_time = utility_functions.get_run_times_mips()
    if joint_featured_map_mode == JointFeatureMode.KRONECKER.value:
        context_dimensions = (
            parametrizations.shape[1] * features.shape[1]
        )
    elif joint_featured_map_mode == JointFeatureMode.CONCATENATION.value:
        context_dimensions = (
            parametrizations.shape[1] + features.shape[1]
        )
    elif joint_featured_map_mode == JointFeatureMode.POLYNOMIAL.value:
        context_dimensions = 4
        for index in range(
            (features.shape[1] + parametrizations.shape[1]) - 2
        ):
            context_dimensions = context_dimensions + 3 + index
    context_matrix = utility_functions.get_context_matrix(
        parametrizations=parametrizations,
        features=features,
        joint_feature_map_mode=joint_featured_map_mode,
        context_feature_dimensions=context_dimensions,
    )
    regrets = np.zeros((reps, features.shape[0]))
    execution_times = np.zeros(reps)

    def job_producer():
        for algorithm_class in algorithms:
            algorithm_name = algorithm_class.__name__
            parameters = {
                "joint_featured_map_mode": joint_featured_map_mode,
                "solver": solver,
                "subset_size": subset_size,
                "parametrizations": parametrizations,
                "features": features,
                "running_time": running_time,
                "context_dimensions": context_dimensions,
                "context_matrix": context_matrix
            }
            for rep_id in range(reps):
                random_state = np.random.RandomState(
                    (base_random_seed + hash(algorithm_name) + rep_id) % 2**32
                )
                yield delayed(single_experiment)(
                    random_state,
                    algorithm_class,
                    algorithm_name,
                    parameters,
                    rep_id,
                )

    jobs = list(job_producer())
    result = Parallel(n_jobs=n_jobs, verbose=10)(jobs)
    runtime = perf_counter() - start_time
    result_df = pd.concat(result)
    algorithm_name = result_df["algorithm"].unique()
    print("Saving files...")
    for name in algorithm_name:
        for rep_id in range(reps):
            mask = (result_df["algorithm"] == name) & (result_df["rep_id"] == rep_id)
            regrets[rep_id] = result_df[mask]["regret"]
            execution_times[rep_id] = result_df[mask]["execution_time"].mean()
        np.save(f"Regret_results_theta0_50//regret_{name}_{solver}_{subset_size}", regrets)
        np.save(
            f"Execution_times_results_theta0_50//execution_time_{name}_{solver}_{subset_size}",
            execution_times,
        )
    print(f"Experiments took {round(runtime)}s.")


def single_experiment(
    task_random_state,
    algorithm_class,
    algorithm_name,
    parameters,
    rep_id,
):
    solver = parameters["solver"]
    subset_size = parameters["subset_size"]

    print(f"{algorithm_name} with {solver} and {subset_size} started...")

    parameters["random_state"] = task_random_state
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
            "regret": regret,
            "execution_time": execution_time,
        }
    )
    return data_frame


if __name__ == "__main__":
    _main()
