import logging
from typing import Optional

import numpy as np

from algorithms.algorithm import Algorithm
from util.constants import JointFeatureMode, Solver


class IndependentSelfSparring(Algorithm):
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
        learning_rate: float = 1,
        logger_name="IndependentSelfSparring",
        logger_level=logging.INFO,
    ) -> None:
        super().__init__(
            random_state=random_state,
            joint_featured_map_mode=joint_featured_map_mode,
            solver=solver,
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
        self.logger.info("Initializing...")
        
        self.learning_rate = learning_rate
        self.wins = self.preference_estimate.wins
        self.losses = self.preference_estimate.losses
        
    def step(self) -> None:
        context_vector = self.context_matrix[self.time_step - 1]  
        self.theta_hat = np.zeros((self.num_arms, self.context_dimensions))
        pairwise_feedback_matrix = np.zeros((self.subset_size, self.subset_size))
        # -----------------WITH CONTEXT----------------
        for i in range(self.num_arms):
            self.theta_hat[i] = self.random_state.beta(
                self.wins[i] + 1, self.losses[i] + 1, size=self.context_dimensions
            )
        self.theta_hat = np.mean(self.theta_hat, axis=0)
        self.skill_vector[self.time_step - 1] = np.mean(
            np.exp(np.inner(self.theta_hat, context_vector)), axis=0
        )
        

        self.selection = self.get_selection(
            quality_of_arms=self.skill_vector[self.time_step - 1]
        )
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        for index_j, j in enumerate(self.selection):
            for index_k, k in enumerate(self.selection):
                if j == self.winner:
                    pairwise_feedback_matrix[index_j][index_k] = 1
                if j == k:
                    pairwise_feedback_matrix[index_j][index_k] = np.nan
                
        
        for index_j, j in enumerate(self.selection):
            for index_k, k in enumerate(self.selection):
                if not np.isnan(pairwise_feedback_matrix[index_j][index_k]):
                    self.wins[j] += self.learning_rate * pairwise_feedback_matrix[index_j][index_k]
                    self.losses[j] += self.learning_rate * (1 - pairwise_feedback_matrix[index_j][index_k])
            
        self.compute_regret(selection=self.selection, time_step=self.time_step)

class IndependentSelfSparringContextual(Algorithm):
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
        gaussian_constant: Optional[float] = 1,
        epsilon: Optional[float] = None,
        failure_probability: Optional[float] = 0.9,
        learning_rate: float = 1,
        logger_name="IndependentSelfSparringContextual",
        logger_level=logging.INFO,
    ) -> None:
        super().__init__(
            random_state=random_state,
            joint_featured_map_mode=joint_featured_map_mode,
            solver=solver,
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
        self.logger.info("Initializing...")
        
        self.learning_rate = learning_rate

        if epsilon is None:
            epsilon = 1/ np.log(self.time_horizon + 1)
        self.B = np.ones((self.num_arms, self.context_dimensions))
        self.mu_hat = np.zeros((self.num_arms, self.context_dimensions))
        self.f = np.zeros((self.num_arms, self.context_dimensions))
        self.v = gaussian_constant * np.sqrt(
            np.divide(24, epsilon)
            * self.context_dimensions
            * np.log(np.divide(1, failure_probability))
        )
        self.context_vector = None
        
    def step(self) -> None:
        standard_deviation = np.divide(self.v**2, self.B)
        self.context_vector = self.context_matrix[self.time_step - 1]
        theta = self.random_state.normal(self.mu_hat, standard_deviation)
        skill_vector = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            skill_vector[arm] = np.exp(np.inner(theta[arm, :], self.context_vector[arm, :]))
        
        self.skill_vector[self.time_step - 1] = skill_vector
        

        self.selection = self.get_selection(
            quality_of_arms=self.skill_vector[self.time_step - 1]
        )
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.preference_estimate.enter_sample(winner_arm=self.winner)
                
        self.compute_regret(selection=self.selection, time_step=self.time_step)
        for arm in self.selection:
            self.B[arm] += self.learning_rate * np.inner(self.context_vector[arm, :], self.context_vector[arm, :])
            if arm == self.winner: 
                self.f[arm] += self.learning_rate * self.context_vector[arm, :]
            else:
                self.f[arm] += 0
            self.mu_hat[arm] = np.divide(self.f[arm], self.B[arm])