import logging
import sys

import numpy as np

from util import utility_functions

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s | %(name)s (%(levelname)s):\t %(message)s"
)
from util.constants import JointFeatureMode

repitition_num = 10
parametrizations = utility_functions.get_parameterization_saps()
features = utility_functions.get_features_saps()
running_time = utility_functions.get_run_times_saps()
context_dimensions = 4
for index in range((features.shape[1] + parametrizations.shape[1]) - 2):
    context_dimensions = context_dimensions + 3 + index

context_matrix = utility_functions.get_context_matrix(
    parametrizations=parametrizations,
    features=features,
    joint_feature_map_mode=JointFeatureMode.POLYNOMIAL.value,
    context_feature_dimensions=context_dimensions,
)

import matplotlib.pyplot as plt
from util.metrics import compute_cumulative_regret
def plot_ts(regret_ts, size):
    cum_reg = compute_cumulative_regret(regret_ts)
    cum_reg_ucb = compute_cumulative_regret(np.load(f"Regret_results_theta0_feedback_correction_50/regret_UCB_saps_{size}.npy", allow_pickle=True))
    cum_reg_colstim = compute_cumulative_regret(np.load(f"Regret_results_theta0_feedback_correction_50/regret_Colstim_saps_{size}.npy", allow_pickle=True))
    cum_reg_colstim_v2 = compute_cumulative_regret(np.load(f"Regret_results_theta0_feedback_correction_50/regret_Colstim_v2_saps_{size}.npy", allow_pickle=True))
    plt.plot(np.mean(cum_reg, axis=0), color='brown', label="Thompson Sampling Context")
    plt.plot(np.mean(cum_reg_ucb, axis=0), color='red', label="UCB")
    plt.plot(np.mean(cum_reg_colstim, axis=0), color='blue', label="Colstim")
    plt.plot(np.mean(cum_reg_colstim_v2, axis=0), color='green', label="Colstim_v2")
    plt.title(f"k = {size}")
    plt.legend()
    plt.show()
    
regret_ts = np.zeros((repitition_num, features.shape[0]))
execution_time_ts = np.zeros(repitition_num)
from algorithms import ThompsonSamplingContextual

for size in [5, 6, 7, 8, 9, 10, 16]:
    for rep in range(repitition_num):
        print(f"Rep no.: {rep + 1}")
        ts = ThompsonSamplingContextual(
            random_state=np.random.RandomState(515),
            parametrizations=parametrizations,
            features=features,
            running_time=running_time,
            context_dimensions=context_dimensions,
            context_matrix=context_matrix,
            subset_size=size,
        )
        ts.run()
        regret_ts[rep] = ts.get_regret()
        execution_time_ts[rep] = ts.execution_time
    plot_ts(regret_ts, size)
