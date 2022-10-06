import multiprocessing
from typing import Optional

import numpy as np
from stats.confidence_radius import HoeffdingConfidenceRadius
from stats.preference_estimate import PreferenceEstimate
from util import utility_functions

from algorithms.interfaces import PacAlgorithm


class TraceTheBest(PacAlgorithm):
    """_summary_

    Parameters
    ----------
    PacAlgorithm : _type_
        _description_
    """

    def __init__(
        self,
        skill_vector,
        subset_size: Optional[int] = multiprocessing.cpu_count(),
        error_bias: float = 0.1,  # epsilon
        failure_probability: float = 0.05,  # delta
    ):
        super().__init__(skill_vector=skill_vector)
        self.subset_size = subset_size
        num_arms = self.feedback_mechanism.get_num_arms()

        try:
            assert (
                0 < error_bias <= 0.5
            ), f"error bias must have values: 0 < error_bias <= 0.5, given {error_bias}"
        except AssertionError:
            raise

        self.error_bias = error_bias

        try:
            assert (
                0 < failure_probability <= 1
            ), f"confidence parameter must have values:0 < failure_probability <= 1, given {failure_probability}"
        except AssertionError:
            raise

        self.failure_probability = failure_probability

        def probability_scaling(num_samples: int) -> float:
            return 4 * (num_arms * num_samples) ** 2

        self.preference_estimate = PreferenceEstimate(
            num_arms=num_arms,
            confidence_radius=HoeffdingConfidenceRadius(
                failure_probability, probability_scaling
            ),
        )
        self.running_winner: int = None  # r_l
        self.running_winner = self.random_state.choice(
            self.feedback_mechanism.get_arms(), replace=False
        )
        self.actions = utility_functions.random_sample_from_list(
            array=self.feedback_mechanism.get_arms(),
            random_state=self.random_state,
            exclude=self.running_winner,
            size=self.subset_size - 1,
        )  # A
        self.actions = np.append(self.actions, self.running_winner)
        self.subset = utility_functions.exclude_elements(
            array=self.feedback_mechanism.get_arms(), exclude=self.actions
        )  # S

        self.rounds = None
        self.empirical_winner: int = None  # c_l

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
                            first_arm_index=i,
                            second_arm_index=j,
                            first_won=self.feedback_mechanism.duel(i, j),
                        )
            self.rounds = self.rounds - 1
        wins = dict()
        for arm in self.actions:
            wins[arm] = self.preference_estimate.get_wins(arm_index=arm)
        self.empirical_winner = max(wins, key=wins.get)

        if self.preference_estimate.get_pairwise_preference_score(
            first_arm_index=self.empirical_winner, second_arm_index=self.running_winner
        ) > (1 / 2) + (self.error_bias / 2):
            prev_running_winner = self.running_winner
            self.running_winner = self.empirical_winner
        else:
            prev_running_winner = self.running_winner

        if self.subset.size < self.subset_size - 1:
            self.actions = utility_functions.random_sample_from_list(
                array=self.actions,
                random_state=self.random_state,
                exclude=prev_running_winner,
                size=(self.subset_size - 1 - self.subset.size),
            )
            self.actions = np.append(
                np.append(self.actions, prev_running_winner), self.subset
            )
            self.subset = np.array([])
        else:
            self.actions = utility_functions.random_sample_from_list(
                array=self.subset,
                random_state=self.random_state,
                size=(self.subset_size - 1),
            )
            self.actions = np.append(self.actions, prev_running_winner)
            self.subset = utility_functions.exclude_elements(
                array=self.subset, exclude=self.actions
            )

    def is_finished(self) -> bool:
        if self.subset.size == 0:
            return True
        else:
            return False

    def get_condorcet_winner(self):
        winner = self.running_winner
        return winner
