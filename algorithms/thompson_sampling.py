from cProfile import run
import logging
import numpy as np
from typing import Optional
from algorithms import Algorithm
from feedback.multi_duel_feedback import MultiDuelFeedback
from util.constants import JointFeatureMode, Solver


class ThompsonSampling(Algorithm):
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        omega: Optional[float] = None,
        subset_size: Optional[int] = ...,
        parametrizations: Optional[np.array] = None,
        features: Optional[np.array] = None,
        running_time: Optional[np.array] = None,
        logger_name="ThompsonSampling",
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
            running_time,
            logger_name,
            logger_level,
        )

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.logger.info("Initializing...")
        # self.theta_hat = np.zeros((self.num_arms, self.context_dimensions))
    
    def step(self):
        context_vector = self.context_matrix[self.time_step - 1]
        wins = self.preference_estimate.wins
        losses = self.preference_estimate.losses
        
        # -----------------WITH CONTEXT----------------
        # for i in range(self.num_arms):
        #     self.theta_hat[i] = self.random_state.beta(wins[i] + 1, losses[i] + 1, size=self.context_dimensions)
        # self.skill_vector[self.time_step - 1] = np.mean(np.exp(np.inner(self.theta_hat, context_vector)), axis=0)
        # self.confidence[self.time_step - 1] = self.get_confidence_bounds(
        #     selection=self.selection,
        #     time_step=self.time_step,
        #     context_vector=context_vector,
        #     winner=self.winner,
        # )
        
        # -----------------WITHOUT CONTEXT----------------
        self.skill_vector[self.time_step - 1] = np.exp(self.random_state.beta(wins + 1, losses + 1))
        
        self.selection = self.get_selection(quality_of_arms=self.skill_vector[self.time_step - 1])
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        self.compute_regret(selection=self.selection, time_step=self.time_step)
