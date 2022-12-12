import logging
import sys

import numpy as np

from util import utility_functions

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s | %(name)s (%(levelname)s):\t %(message)s"
)
from util.constants import JointFeatureMode

repitition_num = 10
parametrizations = utility_functions.get_parameterization_mips()
features = utility_functions.get_features_mips()
running_time = utility_functions.get_run_times_mips()
context_dimensions = 4
for index in range((features.shape[1] + parametrizations.shape[1]) - 2):
    context_dimensions = context_dimensions + 3 + index

context_matrix = utility_functions.get_context_matrix(
    parametrizations=parametrizations,
    features=features,
    joint_feature_map_mode=JointFeatureMode.POLYNOMIAL.value,
    context_feature_dimensions=context_dimensions,
)
regret_ucb = np.zeros((repitition_num, features.shape[0]))
execution_time_ucb = np.zeros(repitition_num)
from algorithms import UCB, Colstim_v2

for rep in range(repitition_num):
    print(f"Rep no.: {rep + 1}")
    ucb = Colstim_v2(
        random_state=np.random.RandomState(515),
        parametrizations=parametrizations,
        features=features,
        running_time=running_time,
        context_dimensions=context_dimensions,
        context_matrix=context_matrix,
        subset_size=5,
    )
    ucb.run()
    regret_ucb[rep] = ucb.get_regret()
    execution_time_ucb[rep] = ucb.execution_time
    # np.save('regret_UCB_saps_16', regret_ucb)
