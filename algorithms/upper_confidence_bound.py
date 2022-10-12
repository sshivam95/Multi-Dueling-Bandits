import logging
import multiprocessing
import secrets
from time import perf_counter
from typing import Optional

import numpy as np
from feedback.multi_duel_feedback import MultiDuelFeedback
from stats.preference_estimate import PreferenceEstimate
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
            parametrizations=parametrizations,
            features=features,
            running_time=running_time,
            subset_size=subset_size,
            logger_level=logger_level
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.feedback_mechanism = MultiDuelFeedback(num_arms=self.num_arms)
        self.preference_estimate = PreferenceEstimate(num_arms=self.num_arms)
        self.time_step = 0
        self.selection = self.random_state.choice(
            self.num_arms, self.subset_size, replace=False
        )  # start with random selection
        self.logger.debug(f"    -> Initial Selection: {self.selection}")
        self.winner = None
        self.execution_time = 0
        self.temp_selec = list()

    def run(self):
        self.logger.info("Running algorithm...")
        start_time = perf_counter()
        
        for self.time_step in range(1, self.time_horizon + 1):
            self.step()
        
        end_time = perf_counter()
        self.execution_time = end_time - start_time
        print("Execution time: ", self.execution_time)
        self.logger.info("Algorithm Finished...")

    def step(self):
        self.logger.debug(f"    -> Time Step: {self.time_step}")
        context_vector = self.context_matrix[self.time_step - 1]        
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            context_vector=context_vector
        )        
        self.confidence[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection, time_step=self.time_step, context_vector=context_vector, winner=self.winner
        )
        quality_of_arms = (
            self.skill_vector[self.time_step - 1] + self.confidence[self.time_step - 1]
        )
        self.selection = self.get_selection(
            quality_of_arms=quality_of_arms
        )
        self.temp_selec.append(self.selection)
        self.logger.debug(f"    -> Selection: {self.selection}")
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )        
        self.logger.debug(f"    -> Selection Winner: {self.winner}")
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        self.update_theta(
            selection=self.selection, time_step=self.time_step, winner=self.winner
        )
        self.compute_regret(selection=self.selection, time_step=self.time_step)
