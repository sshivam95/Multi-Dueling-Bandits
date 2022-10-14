import logging
import sys

import numpy as np

from algorithms import Colstim
from util import utility_functions

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s | %(name)s (%(levelname)s):\t %(message)s"
)


def _main():
    repitition_num = 50
    parametrizations = utility_functions.get_parameterization_saps()
    features = utility_functions.get_features_saps()
    running_time = utility_functions.get_run_times_saps()

    for rep in range(repitition_num):
        print(f"Rep no.: {rep + 1}")
        regret_colstim = np.zeros((repitition_num, features.shape[0]))
        execution_time_colstim = np.zeros(repitition_num)
        colstim = Colstim(
            random_state=np.random.RandomState(515),
            parametrizations=parametrizations,
            features=features,
            running_time=running_time,
            # logger_level=logging.DEBUG,
        )
        colstim.run()
        regret_colstim[rep] = colstim.get_regret()
        execution_time_colstim[rep] = colstim.execution_time

    np.savetxt("regret_colstim_saps.txt", regret_colstim)
    np.savetxt("execution_time_colstim_saps.txt", execution_time_colstim)


if __name__ == "__main__":
    _main()
