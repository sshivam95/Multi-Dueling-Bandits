from gc import is_finalized
import multiprocessing
from typing import Optional
import numpy as np

from algorithms.interfaces import PacAlgorithm
from stats.confidence_radius import HoeffdingConfidenceRadius
from stats.preference_estimate import PreferenceEstimate

from util import utility_functions


class TraceTheBest(PacAlgorithm):
    def __init__(
        self,
        time_horizon: Optional[int] = None,
        subset_size: Optional[int] = multiprocessing.cpu_count(),
        error_bias: Optional[float] = None, # epsilon
        failure_probability: Optional[float] = None, # delta
    ):
        super().__init__(time_horizon)
        self.subset_size = subset_size
        num_arms = self.feedback_mechanism.get_num_arms()

        if error_bias is not None:
            try:
                assert (
                    0 < error_bias <= 0.5
                ), f"error bias must have values: 0 < error_bias <= 0.5, given {error_bias}"
            except AssertionError:
                raise
        else:
            error_bias = self.random_state.uniform(0, 0.5)
        self.error_bias = error_bias

        if failure_probability is not None:
            try:
                assert (
                    0 < failure_probability <= 1
                ), f"confidence parameter must have values:0 < failure_probability <= 1, given {failure_probability}"
            except AssertionError:
                raise
        else:
            failure_probability = self.random_state.uniform(0, 1)
        self.failure_probability = failure_probability
        
        def probability_scaling(num_samples: int) -> float:
            return 4 * (num_arms * num_samples) ** 2

        self.preference_estimate = PreferenceEstimate(
            num_arms=num_arms,
            confidence_radius=HoeffdingConfidenceRadius(failure_probability, probability_scaling),
        ) 
        self.running_winner = list() # r_l
        self.running_winner.append(
            self.random_state.choice(self.feedback_mechanism.get_arms(), replace=False)
        )
        self.actions = utility_functions.random_sample_from_list(
            array=self.feedback_mechanism.get_arms(),
            random_state=self.random_state,
            exclude=self.running_winner,
            size=self.subset_size - 1,
        ) # A
        self.actions = np.append(self.actions, self.running_winner)
        self.subset = utility_functions.exclude_elements(
            array=self.feedback_mechanism.get_arms(), exclude=self.actions
        ) # S

        self.rounds = None
        self.empirical_winner = list() # c_l

    def run(self) -> None:
        while not self.is_finished():
            self.step()

    def step(self) -> None:
        self.rounds = np.divide(2 * self.subset_size, self.error_bias**2) * np.log(
            np.divide(
                2 * self.feedback_mechanism.get_num_arms(), self.failure_probability
            )
        )
        while self.rounds > 0:
            for i in self.actions:
                for j in self.actions:
                    if i != j:
                        self.preference_estimate.enter_sample(
                            first_arm_index=i, second_arm_index=j, first_won=self.feedback_mechanism.duel(i, j)
                        )
            self.rounds =- 1
        # self.empirical_winner.append(utility_functions.argmax_set(self.preference_estimate.wins))

    def _update_confidence_radius(self) -> None:
        confidence_radius = HoeffdingConfidenceRadius()
        self.preference_estimate.set_confidence_radius(confidence_radius())

    def is_finished(self) -> bool:
        return False


    