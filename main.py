import logging
import sys

import numpy as np

from algorithms import UCB
from util import utility_functions

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s | %(name)s (%(levelname)s):\t %(message)s"
)


def _main():
    repitition_num = 1
    parametrizations = utility_functions.get_parameterization_saps()
    features = utility_functions.get_features_saps()
    running_time = utility_functions.get_run_times_saps()
    regret_ucb = np.zeros((repitition_num, features.shape[0]))
    execution_time_ucb = np.zeros(repitition_num)

    for rep in range(repitition_num):
        print(f"Rep no.: {rep + 1}")
        ucb = UCB(
            random_state=np.random.RandomState(515),
            parametrizations=parametrizations,
            features=features,
            running_time=running_time,
            logger_level=logging.DEBUG,
        )
        ucb.run()
        regret_ucb[rep] = ucb.get_regret()
        execution_time_ucb[rep] = ucb.execution_time
    pass


if __name__ == "__main__":
    _main()
