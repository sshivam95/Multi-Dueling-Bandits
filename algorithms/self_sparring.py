"""An implementation of the Independent Self Sparring algorithm strategy."""
import logging
from typing import Optional

import numpy as np

from algorithms.algorithm import Algorithm
from util.constants import JointFeatureMode, Solver


class IndependentSelfSparring(Algorithm):
    """The independent self sparring uses Beta-Bernoulli Thompson sampling as a subroutine to independently sample the set of `k` arms to duel. 
    
    The algorithm uses `k`  number of stochastic MAB algorithms (in this case Beta-Bernoulli Thompson Sampling) to control the choice of each arm.
    Although the original algorithm does not use the contextual information when sampling the arms.
    However, in this class, we implements the Beta-Bernoulli Thompson sampling with contextual information to match the CPPL framework.
    
    The original algorithm used is Algorithm 3 from paper by Sui et. al. (2017) https://arxiv.org/pdf/1705.00253.pdf

    Parameters
    ----------
    learning_rate : float
        Corresponds to `eta` in the paper.
        
    Attributes
    ----------
    wins: np.array
        The number of wins of each arm in the pool. A win for am arm is considered when it has the lowest run time than the other arms in terms of solving the problem instance.
    losses: np.array
        The number of losses of each arm in the pool.
    """
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
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
        # Observe the context vector
        context_vector = self.context_matrix[self.time_step - 1]
        
        # Since we are sampling the estimated latent mean utility parameter, theta_hat, from the perboulli distribution at each time step, we have to initialize theta_hat for each time step. 
        # The algorithm learns the probability distribution based on the number of wins and losses of the arms.
        self.theta_hat = np.zeros((self.num_arms, self.context_dimensions))
        
        # A pairwise feedback matrix is initialized of size `subset_size` with `np.nan`.
        pairwise_feedback_matrix = np.empty((self.subset_size, self.subset_size))
        pairwise_feedback_matrix.fill(np.nan)
        
        # Sample theta_hat using Beta distribution
        for i in range(self.num_arms):
            self.theta_hat[i] = self.random_state.beta(
                self.wins[i] + 1, self.losses[i] + 1, size=self.context_dimensions
            )
        
        # Calculate the contextualized skill vector.
        for i in range(self.num_arms):
            self.skill_vector[self.time_step - 1][i] = np.exp(np.inner(context_vector[i], self.theta_hat[i]))

        # Get the subset based on the skill vector. 
        # Since Thompson Sampling learns from a probability distribution, no confidence bounds were used.
        self.selection = self.get_selection(
            quality_of_arms=self.skill_vector[self.time_step - 1]
        )
        
        # Get the winner feedback from the feedback mechanism
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        
        # Update the samples and the pairwise feedback matrix
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        for index_j, j in enumerate(self.selection):
            for index_k, k in enumerate(self.selection):
                if j == self.winner:
                    pairwise_feedback_matrix[index_j][index_k] = 1
                if j == k:
                    pairwise_feedback_matrix[index_j][index_k] = 0

        # Update the wins and losses accordingly.
        for index_j, j in enumerate(self.selection):
            for index_k, k in enumerate(self.selection):
                if not np.isnan(pairwise_feedback_matrix[index_j][index_k]):
                    self.wins[j] += self.learning_rate * pairwise_feedback_matrix[index_j][index_k]
                    self.losses[j] += self.learning_rate * (1 - pairwise_feedback_matrix[index_j][index_k])

        # Compute the regret
        self.compute_regret(selection=self.selection, time_step=self.time_step)

class IndependentSelfSparringContextual(Algorithm):
    """This version of independent self sparring uses Thompson Sampling for contextual as a subroutine. 
    
    The subroutine used in this version used is from Algorithm 1 by Agarwal et. al. (2013) http://proceedings.mlr.press/v28/agrawal13.pdf with a combination of sampling `k` arms.
    The Thompson Sampling subroutine uses a Guassian likelihodd function and Gaussian prior to sample the utility weight parameters.

    Parameters
    ----------
    gaussian_constant: float, optional
        A constant used in calculating a component of the probability density function of the  Gaussian distribution, by default is 1.
    epsilon: float, optional
        The optimality of the winning arm. The value ranges from (0, 1), by default is None.
    failure_probability: float, optional
        The probability that the result is not correct. Corresponds to `delta` in the subroutine algorithm, by default is 0.01
    learning_rate : float
        Corresponds to `eta` in the Sui et. al. (2017) paper.
        
    Attributes
    ----------
    B: np.array
        A component of the probability density function of Gaussian distribution.
    v: float
        A component of the probability density function of Gaussian distribution.
    """
    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        subset_size: Optional[int] = ...,
        parametrizations: Optional[np.array] = None,
        features: Optional[np.array] = None,
        context_matrix: Optional[np.array] = None,
        context_dimensions: Optional[int] = None,
        running_time: Optional[np.array] = None,
        gaussian_constant: Optional[float] = 1,
        epsilon: Optional[float] = None,
        failure_probability: Optional[float] = 0.01,
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
        self.B = np.ones((self.context_dimensions, self.context_dimensions))
        self.theta_hat = np.zeros((self.context_dimensions, self.context_dimensions))
        self.f = np.zeros((self.context_dimensions, self.context_dimensions))
        self.v = gaussian_constant * np.sqrt(
            np.divide(24, epsilon)
            * self.context_dimensions
            * np.log(np.divide(1, failure_probability))
        )
        self.context_vector = None

    def step(self) -> None:
        # Observe the context vector of the current time step
        self.context_vector = self.context_matrix[self.time_step - 1]
        
        # A pairwise feedback matrix is initialized of size `subset_size` with `np.nan`
        pairwise_feedback_matrix = np.empty((self.subset_size, self.subset_size))
        pairwise_feedback_matrix.fill(np.nan)
        
        # Sample theta using the probability density function of the Gaussian distribution
        try:
            self.B_inv = np.linalg.inv(self.B).astype("float64")
        except np.linalg.LinAlgError as error:
            self.B_inv = np.abs(np.linalg.pinv(self.B).astype("float64"))
        standard_deviation = np.inner(self.v**2, self.B_inv)
        theta = self.random_state.normal(self.theta_hat, standard_deviation)
        
        # Compute the skill vector using the sampled theta and the context vector
        skill_vector = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            skill_vector[arm] = np.exp(np.inner(theta[arm, :], self.context_vector[arm, :]))
        self.skill_vector[self.time_step - 1] = skill_vector

        # Select the subset based on the skill vector
        self.selection = self.get_selection(
            quality_of_arms=self.skill_vector[self.time_step - 1]
        )
        
        # Get the winner feedback from the feedback mechanism
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        
        # Update the samples and the pairwise feedback matrix
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        for index_j, j in enumerate(self.selection):
            for index_k, k in enumerate(self.selection):
                if j == self.winner:
                    pairwise_feedback_matrix[index_j][index_k] = 1
                if j == k:
                    pairwise_feedback_matrix[index_j][index_k] = 0
        
        # Update other stats
        for arm_index, arm in enumerate(self.selection):
            self.B += self.learning_rate * np.inner(self.context_vector[arm, :], self.context_vector[arm, :])
            if arm == self.winner:
                self.f += self.learning_rate * self.context_vector[arm, :] * np.max(pairwise_feedback_matrix[arm_index, :])
            else:
                self.f += 0
        try:
            self.B_inv = np.linalg.inv(self.B).astype("float64")
        except np.linalg.LinAlgError as error:
            self.B_inv = np.linalg.pinv(self.B).astype("float64")
        self.theta_hat = np.inner(self.B_inv, self.f)
        
        # Compute regret
        self.compute_regret(selection=self.selection, time_step=self.time_step)
