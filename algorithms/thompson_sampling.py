"""An implementation of the Thompson sampling algorithm for arm selection strategy."""
import logging
import warnings
from typing import Optional

# suppress warnings
warnings.filterwarnings("ignore")
import numpy as np

from algorithms import Algorithm
from util.constants import JointFeatureMode, Solver


class ThompsonSampling(Algorithm):
    """This class implements Beta-Bernoulli Thompson Sampling algorithm which is used as a sub-routine for the independent self-sparring algorithm.

    The original algorithm used is Algorithm 1 from paper by Sui et. al. (2017) https://arxiv.org/pdf/1705.00253.pdf.
    Although the original algorithm does not use the contextual information when sampling the arms.
    However, in this class, we implements the Beta-Bernoulli Thompson sampling with contextual information to match the CPPL framework.
    """

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
            context_matrix,
            context_dimensions,
            running_time,
            logger_name,
            logger_level,
        )

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.logger.info("Initializing...")

    def step(self):
        # Observe the contextual information for the current time step
        context_vector = self.context_matrix[self.time_step - 1]
        wins = self.preference_estimate.wins
        losses = self.preference_estimate.losses
        self.theta_hat = np.zeros((self.num_arms, self.context_dimensions))

        # Sample the estimated utility parameter from the Beta distribution
        for i in range(self.num_arms):
            self.theta_hat[i] = self.random_state.beta(
                wins[i] + 1, losses[i] + 1, size=self.context_dimensions
            )

        # Compute the skill vector
        for i in range(self.num_arms):
            self.skill_vector[self.time_step - 1][i] = np.exp(
                np.inner(context_vector[i], self.theta_hat[i])
            )

        # Get the subset selection based on the skill vector
        self.selection = self.get_selection(
            quality_of_arms=self.skill_vector[self.time_step - 1]
        )

        # Get the winner feedback from the feedback mechanism
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )

        # Update the samples
        self.preference_estimate.enter_sample(winner_arm=self.winner)

        # Compute the regret
        self.compute_regret(selection=self.selection, time_step=self.time_step)


class ThompsonSamplingContextual(ThompsonSampling):
    """This version of Thompson Sampling is for contextual bandits combined with the multi-dueling bandits.

    This version used is from Algorithm 1 by Agarwal et. al. (2013) http://proceedings.mlr.press/v28/agrawal13.pdf with a combination of sampling `k` arms.
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
        omega: Optional[float] = None,
        subset_size: Optional[int] = ...,
        parametrizations: Optional[np.array] = None,
        features: Optional[np.array] = None,
        context_matrix: Optional[np.array] = None,
        context_dimensions: Optional[int] = None,
        running_time: Optional[np.array] = None,
        gaussian_constant: Optional[float] = 1,
        epsilon: Optional[float] = None,
        failure_probability: Optional[float] = 0.01,
        logger_name="ThompsonSamplingContextual",
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
            logger_level=logger_level,
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.logger.info("Initializing...")

        if epsilon is None:
            epsilon = 1 / np.log(self.time_horizon + 1)
        self.B = np.ones((self.context_dimensions, self.context_dimensions))
        self.mu_hat = np.zeros((self.context_dimensions, self.context_dimensions))
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

        # Sample theta using the probability density function of the Gaussian distribution
        try:
            self.B_inv = np.linalg.inv(self.B).astype("float64")
        except np.linalg.LinAlgError as error:
            self.B_inv = np.abs(np.linalg.pinv(self.B).astype("float64"))
        standard_deviation = np.inner(self.v**2, self.B_inv)
        theta = self.random_state.normal(self.mu_hat, np.abs(standard_deviation))

        # Compute the skill vector using the sampled theta and the context vector
        skill_vector = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            skill_vector[arm] = np.exp(
                np.inner(theta[arm, :], self.context_vector[arm, :])
            )
        self.skill_vector[self.time_step - 1] = skill_vector

        # Select the subset based on the skill vector
        self.selection = self.get_selection(
            quality_of_arms=self.skill_vector[self.time_step - 1]
        )

        # Get the winner feedback from the feedback mechanism
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )

        # Update the samples
        self.preference_estimate.enter_sample(winner_arm=self.winner)

        # Update other stats
        self.update()

        # Compute regret
        self.compute_regret(selection=self.selection, time_step=self.time_step)

    def update(self) -> None:
        """Update the statistics of the Thampson Sampling for Contextual Bandits."""
        for arm in self.selection:
            self.B += np.inner(self.context_vector[arm, :], self.context_vector[arm, :])
            if arm == self.winner:
                self.f += self.context_vector[arm, :]
            else:
                self.f += 0
        try:
            self.B_inv = np.linalg.inv(self.B).astype("float64")
        except np.linalg.LinAlgError as error:
            self.B_inv = np.linalg.pinv(self.B).astype("float64")
        self.mu_hat = np.inner(self.B_inv, self.f)
