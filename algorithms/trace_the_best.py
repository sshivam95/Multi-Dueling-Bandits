import enum
import logging
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
        rounds: Optional[int] = None,
        subset_size: Optional[int] = multiprocessing.cpu_count(),
        error_bias: float = 0.1,  # epsilon
        failure_probability: float = 0.05,  # delta
        logger_name="TraceTheBest",
        logger_level=logging.INFO,
    ):
        super().__init__()
        self.subset_size = subset_size
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

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

        self.preference_estimate = PreferenceEstimate(num_arms=self.num_arms)
        self.running_winner: int = None  # r_l
        self.running_winner = self.random_state.choice(self.get_arms(), replace=False)
        self.actions = utility_functions.random_sample_from_list(
            array=self.get_arms(),
            random_state=self.random_state,
            exclude=self.running_winner,
            size=self.subset_size - 1,
        )  # A
        self.actions = np.append(self.actions, self.running_winner)
        self.subset = utility_functions.exclude_elements(
            array=self.get_arms(), exclude=self.actions
        )  # S

        self.rounds = rounds
        self.empirical_winner: int = None  # c_l
        # self.rounds = int(
        #     np.divide(2 * self.subset_size, self.error_bias**2)
        #     * np.log(
        #         np.divide(
        #             2 * self.get_num_arms(), self.failure_probability
        #         )
        #     )
        # )
        if self.rounds > self.time_horizon:
            self.rounds = self.time_horizon
        self.regret = list()
        self.counter = 0
        self.finished = False
        instances = [i for i, _ in enumerate(self.problem_instances)]
        self.round_set = {
            i: i + (self.rounds - 1) for i in instances if i % self.rounds == 0
        }
        self.round_instances = list()
        for k, v in self.round_set.items():
            self.round_instances.append(instances[k:v])

    def run(self) -> None:
        self.logger.info("Running algorithm...")
        while not self.is_finished():
            self.step()
            self.counter = self.counter + 1

    def step(self) -> None:
        self.logger.info(f"Iteration: {self.counter}")
        self.logger.debug(f"    -> Running winner: {self.running_winner}")
        self.logger.debug(f"    -> Actions: {self.actions}")
        self.logger.debug(f"    -> Subset: {self.subset}")
        self.logger.info("Starting Duels...")

        regret = list()
        for instance in self.round_instances[self.counter]:
            # for i in self.actions:
            winner_round = utility_functions.get_round_winner(
                self.running_time[instance]
            )
            self.preference_estimate.enter_sample(winner_arm=winner_round)
            for i in self.actions:
                for j in self.actions:
                    if i != j:
                        self.preference_estimate.set_pairwise_preference_score(
                            first_arm_index=i,
                            second_arm_index=j,
                            context_matrix=self.context_matrix[instance],
                            theta=self.theta_bar,
                        )
                self.confidence
            regret.append(self.compute_regret(time_step=instance, selection=self.actions))
        self.regret.append(regret)

        self.logger.info("Duels finished...")
        wins = dict()
        for arm in self.actions:
            wins[arm] = self.preference_estimate.get_wins(arm_index=arm)
        self.empirical_winner = max(wins, key=wins.get)
        self.logger.debug(f"    -> Empirical winner: {self.empirical_winner}")

        if self.preference_estimate.get_pairwise_preference_score(
            first_arm_index=self.empirical_winner, second_arm_index=self.running_winner
        ) > (1 / 2) + (self.error_bias / 2):
            self.logger.info("p(c_l, r_l) > 1/2 + ∈/2")
            prev_running_winner = self.running_winner
            self.running_winner = self.empirical_winner
            self.logger.debug(f"    -> New running winner: {self.running_winner}")
        else:
            prev_running_winner = self.running_winner
        if self.subset.size == 0:
            self.logger.info("End of Iterations...")
            self.finished = True
            return
        if self.subset.size < self.subset_size - 1:
            self.logger.info("self.subset.size < self.subset_size - 1")
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

            # TODO: UPDATE THETA

    def is_finished(self) -> bool:
        return self.finished

    def get_condorcet_winner(self):
        winner = self.running_winner
        return winner
