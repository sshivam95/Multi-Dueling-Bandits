import argparse
import inspect
import logging
import sys
import warnings
from time import perf_counter
from typing import Generator

import numpy as np
from joblib import Parallel, delayed

from algorithms import regret_minimizing_algorithms
from util import utility_functions
from util.constants import JointFeatureMode, Solver

warnings.filterwarnings("ignore")

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s | %(name)s (%(levelname)s):\t %(message)s"
)


def _main():

    parser = argparse.ArgumentParser(
        description="Run Multi-Dueling Bandits experiments and plot results."
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
        default=50,
        type=int,
        help="How often to run each algorithm. Results will be averaged. (default: 50)",
    )
    parser.add_argument(
        "--subset-size",
        dest="subset_size",
        default=10,
        type=int,
        help="How many arms the generated subset from original arms should contain. (default: 10)",
    )
    parser.add_argument(
        "--dataset",
        dest="solver",
        default=Solver.SAPS.value,
        type=str,
        help="Which dataset of problem instances to use for experimentation. (default: saps)",
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
    run_experiment(
        algorithms=algorithms,
        reps=args.reps,
        joint_featured_map_mode=args.joint_featured_map_mode,
        base_random_seed=args.base_random_seed,
        n_jobs=args.n_jobs,
    )


def run_experiment(
    algorithms,
    reps,
    joint_featured_map_mode,
    base_random_seed,
    n_jobs,
):
    start_time = perf_counter()

    def job_producer() -> Generator:
        for solver in Solver:
            if solver.value == Solver.SAPS.value:
                parametrizations = utility_functions.get_parameterization_saps()
                features = utility_functions.get_features_saps()
                running_time = utility_functions.get_run_times_saps()
            else:
                parametrizations = utility_functions.get_parameterization_mips()
                features = utility_functions.get_features_mips()
                running_time = utility_functions.get_run_times_mips()
            for algorithm_class in algorithms:
                algorithm_name = algorithm_class.__name__
                for subset_size in [5, 6, 7, 8, 9, 10]:
                    parameters = {
                        "joint_featured_map_mode": joint_featured_map_mode,
                        "solver": solver.value,
                        "subset_size": subset_size,
                        "parametrizations": parametrizations,
                        "features": features,
                        "running_time": running_time,
                    }
                    yield delayed(single_experiment)(
                        base_random_seed,
                        algorithm_class,
                        algorithm_name,
                        parameters,
                        reps,
                    )

    jobs = list(job_producer())
    Parallel(n_jobs=n_jobs, verbose=50)(jobs)
    runtime = perf_counter() - start_time
    print(f"Experiments took {round(runtime)}s.")


def single_experiment(
    base_random_seed,
    algorithm_class,
    algorithm_name,
    parameters,
    reps,
):
    solver = parameters["solver"]
    subset_size = parameters["subset_size"]
    regret = np.zeros((reps, parameters["features"].shape[0]))
    execution_time = np.zeros(reps)
    print(f"{algorithm_name} with {solver} and {subset_size} started...")
    for rep_id in range(reps):
        random_state = np.random.RandomState(
            (base_random_seed + hash(algorithm_name) + rep_id) % 2**32
        )
        parameters["random_state"] = random_state
        parameters_to_pass = dict()
        for parameter in parameters.keys():
            if parameter in inspect.getfullargspec(algorithm_class.__init__)[0]:
                parameters_to_pass[parameter] = parameters[parameter]
        algorithm_object = algorithm_class(**parameters_to_pass)
        algorithm_object.run()
        regret[rep_id] = algorithm_object.get_regret()
        execution_time[rep_id] = algorithm_object.execution_time
    print(f"{algorithm_name} with {solver} and {subset_size} finished...")
    print("Saving files")
    np.save(f"regret_{algorithm_name}_{solver}_{subset_size}", regret)
    np.save(f"execution_time_{algorithm_name}_{solver}_{subset_size}", execution_time)
    
    np.savetxt(f"regret_{algorithm_name}_{solver}_{subset_size}.txt", regret)
    np.savetxt(f"execution_time_{algorithm_name}_{solver}_{subset_size}.txt", execution_time)


if __name__ == "__main__":
    _main()
