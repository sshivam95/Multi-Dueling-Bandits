import logging
import multiprocessing
from time import perf_counter
from typing import Optional

import numpy as np

from algorithms.algorithm import Algorithm
from feedback.multi_duel_feedback import MultiDuelFeedback
from stats.preference_estimate import PreferenceEstimate
from util.constants import Solver, JointFeatureMode


class Colstim(Algorithm):
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        exploration_length: Optional[int] = None,
        threshold_parameter: Optional[float] = None,
        confidence_width: Optional[float] = None,
        subset_size: Optional[int] = multiprocessing.cpu_count(),
        logger_name="COLSTIM",
        logger_level=logging.INFO,
    ) -> None:
        super().__init__(
            random_state=random_state,
            joint_featured_map_mode=joint_featured_map_mode,
            solver=solver,
            subset_size=subset_size,
            logger_level=logger_level,
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.feedback_mechanism = MultiDuelFeedback(num_arms=self.num_arms)
        
        if exploration_length is not None:
            try:
                assert (
                    exploration_length > 0
                ), f"exploration length must be greater than 0, given {exploration_length}"
            except AssertionError:
                raise
            self.exploration_length = exploration_length
        else:
            self.exploration_length = self.context_dimensions * self.num_arms

        if threshold_parameter is not None:
            try:
                assert (
                    threshold_parameter > 0
                ), f"threshold parameter must be greater than 0, given {threshold_parameter}"
            except AssertionError:
                raise
            self.threshold_parameter = threshold_parameter
        else:
            self.threshold_parameter = np.sqrt(
                self.context_dimensions * np.log(self.time_horizon)
            )

        if confidence_width is not None:
            try:
                assert (
                    confidence_width > 0
                ), f"confidence width must be greater than 0, given {confidence_width}"
            except AssertionError:
                raise
            self.confidence_width = confidence_width
        else:
            self.confidence_width = np.sqrt(
                self.context_dimensions * np.log(self.time_horizon)
            )
    
    def step(self):
        self.logger.debug(f"    -> Time Step: {self.time_step}")
        context_vector = self.context_matrix[self.time_step - 1]

        self.theta_hat = utility_functions.stochastic_gradient_descent(
            theta=self.theta_hat,
            gamma_t=gamma_t,
            selection=selection,
            context_vector=context_vector,
            winner=winner,
        )

        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            context_vector=context_vector
        )
        self.logger.debug(
            f"    -> Skill Vector: {self.skill_vector[self.time_step - 1]}"
        )

        self.confidence[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=context_vector,
            winner=self.winner,
        )
        self.logger.debug(f"    -> Confidence: {self.confidence[self.time_step - 1]}")

        quality_of_arms = (
            self.skill_vector[self.time_step - 1] + self.confidence[self.time_step - 1]
        )
        self.logger.debug(f"    -> Quality of arms: {quality_of_arms}")

        self.selection = self.get_selection(quality_of_arms=quality_of_arms)
        self.logger.debug(f"    -> Selection: {self.selection}")

        self.logger.debug("Starting Duels...")
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.logger.debug("Duels finished...")

        self.logger.debug(f"    -> Selection Winner: {self.winner}")
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        self.logger.debug("Updating Theta...")
        self.update_theta(
            selection=self.selection, time_step=self.time_step, winner=self.winner
        )
        self.compute_regret(selection=self.selection, time_step=self.time_step)
