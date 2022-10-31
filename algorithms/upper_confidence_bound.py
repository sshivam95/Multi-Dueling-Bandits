import logging
import multiprocessing
from typing import Optional

import numpy as np
from feedback.multi_duel_feedback import MultiDuelFeedback
from util.constants import JointFeatureMode, Solver

from algorithms.algorithm import Algorithm


class UCB(Algorithm):
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        omega: Optional[float] = None,
        subset_size: Optional[int] = multiprocessing.cpu_count(),
        parametrizations: Optional[np.array] = None,
        features: Optional[np.array] = None,
        running_time: Optional[np.array] = None,
        logger_name="UpperConfidenceBound",
        logger_level=logging.INFO,
    ) -> None:
        super().__init__(
            random_state=random_state,
            joint_featured_map_mode=joint_featured_map_mode,
            solver=solver,
            omega=omega,
            subset_size=subset_size,
            parametrizations=parametrizations,
            features=features,
            running_time=running_time,
            logger_level=logger_level,
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.feedback_mechanism = MultiDuelFeedback(num_arms=self.num_arms)

    def step(self):
        self.logger.debug(f"    -> Time Step: {self.time_step}")
        context_vector = self.context_matrix[self.time_step - 1]
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            theta=self.theta_bar, context_vector=context_vector, exp=True
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
        self.update_estimated_theta(
            selection=self.selection, time_step=self.time_step, winner=self.winner
        )
        self.update_mean_theta(self.time_step)
        self.compute_regret(selection=self.selection, time_step=self.time_step)
