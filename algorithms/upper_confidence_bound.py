import logging
from typing import Optional

import numpy as np

from algorithms.algorithm import Algorithm
from util.constants import JointFeatureMode, Solver


class UCBExploreExploit(Algorithm):
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        omega: Optional[float] = None,
        subset_size: Optional[int] = ...,
        parametrizations: Optional[np.array] = None,
        features: Optional[np.array] = None,
        context_matrix: Optional[np.array] = None,
        context_dimensions: Optional[int] = None,
        running_time: Optional[np.array] = None,
        logger_name="BaseAlgorithm",
        logger_level=logging.INFO,
    ) -> None:
        super().__init__(
            random_state,
            joint_featured_map_mode,
            solver,
            omega,
            subset_size,
            parametrizations,
            features,
            context_matrix,
            context_dimensions,
            running_time,
            logger_name,
            logger_level,
        )

    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        omega: Optional[float] = None,
        subset_size: Optional[int] = ...,
        parametrizations: Optional[np.array] = None,
        features: Optional[np.array] = None,
        context_matrix: Optional[np.array] = None,
        context_dimensions: Optional[int] = None,
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
            context_matrix=context_matrix,
            context_dimensions=context_dimensions,
            running_time=running_time,
            logger_level=logger_level,
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

    def step(self):
        self.logger.debug(f"    -> Time Step: {self.time_step}")
        context_vector = self.context_matrix[self.time_step - 1]
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            theta=self.theta_bar, context_vector=context_vector
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

        self.selection = self.get_selection_framework_v2()

        
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

    def get_selection_framework_v2(self):
        quality_of_arms_exploit = self.skill_vector[self.time_step - 1]
        selection_exploit = self.get_selection(
            quality_of_arms=quality_of_arms_exploit,
            subset_size=self.subset_size,
        )

        mask = np.zeros(self.num_arms, bool)
        mask[selection_exploit] = True
        confidence_temp = self.confidence[self.time_step - 1]
        confidence_temp[mask] = np.nan
        quality_of_arms_explore = confidence_temp
        selection_explore = self.get_selection(
            quality_of_arms=quality_of_arms_explore,
            subset_size=self.subset_size,
        )

        # ToDo: Change parameters of get selection to exclude redundent arms
        if self.subset_size % 2 == 0:
            selection = np.concatenate(
                (
                    selection_explore[0 : int(self.subset_size / 2)],
                    selection_exploit[0 : int(self.subset_size / 2)],
                )
            )
        else:
            selection = np.concatenate(
                (
                    selection_explore[0 : int((self.subset_size - 1) / 2)],
                    selection_exploit[0 : int(((self.subset_size - 1) / 2) + 1)],
                )
            )

        return selection
