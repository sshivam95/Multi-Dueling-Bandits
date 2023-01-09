"""A generic superclass which contains the common functionality required in the implemented arm selection strategy algorithms."""

import copy
import logging
import multiprocessing
from time import perf_counter
from typing import Optional

import numpy as np
from tqdm import tqdm

from feedback.multi_duel_feedback import MultiDuelFeedback
from stats.preference_estimate import PreferenceEstimate
from util import metrics, utility_functions
from util.constants import JointFeatureMode, Solver


class Algorithm:
    """Parent class of all the implemented arm selection strategy algorithms.

    When implementing any new arm strategy algorithm, inherit this parent class to mimic the basic functionality of the CPPL framework.
    The basic functionalities include:
        a. Extracting the instance features, parameter features and running times if not given by the user for SAPS and CPLEX solver dataset used in this work.
        b. Initializing all the variables used in the CPPL framework and those common in the arm selection strategy algorithms.
        c. Calculating and updating the regret of an algorithm.
        d. Initializing a general arm selection method.
        e. Calculating the confidence bounds used by some of the different arm selection algorithms.

    Parameters
    ----------
    random_state: np.random.RandomState, optional
        A numpy random state. Defaults to an unseeded state when not specified.
    joint_featured_map_mode: str, optional
        The feature map mode used on the instance features and parameter features by the CPPL for generating the context matrix, by default `polynomial`.
    solver: str, optional
        The solver used by the CPPL to solve the probem instances, by defaults `saps`.
    omega: float, optional
        The hyperparameter used by the UCB strategy, by defualt None.
    subset_size: int, optional
        The size `k` from the literature representing the number of arms to select as a subset from the pool of arms, by defualt multiprocessing.cpu_count(), i.e., the number of CPU cores in the machine.
    parametrizations: np.array, optional
        The parameters of the solver, by default None.
    features: np.array, optional
        The features of the problem instances to be solved by the solver, by default None.
    context_matrix: np.array, optional
        The context matrix of all the problem instances and all the arms. Usually this is constructed by the `utility_functions`, by default None.
    context_dimensions: int, optional
        The contextual dimensions of the context vector based on the joint feature map mode, by default None.
    running_time: np.array, optional
        The execution times of each parameter sequence, or arms, runned on the instance problems by the solver, by default None.

    Attributes
    ----------
    num_arms
        The total number of arms running in the experiments. Usually this number is equal to the combination of the solver's parameterizations.
    feedback_mechanism
        A `FeedbackMechanism` object describing the environment. This parameter has been taken from the parent class.
    time_horizon
        This is usually the total number of the problem instances given to the CPPL.
    context_dimensions
        The dimension of the context vector of a single arm for a single problem instance. This quantity is determined by the joint feature mode given by the user.
    theta_init
        The initial weight parameter vector for the context vector. Initially, they are randomly initialized (as per the CPPL algorithm) and have a shape of (context_dimensions, 1).
    theta_hat
        The maximum-likelihood estimate of the weight parameter. Initially, it is equal to the `theta_init`.
    theta_bar
        The weight parameter used by the Polyak-Ruppert averaged Stochastic Gradient Descent (SGD) method used in the CPPL framework for certain arm strategies (eg. UCB, Colstim, and Colstim_v2). Initially equal to `theta_hat`.
    regret
        The regret measurement vector of shape (time_horizon, 1) for each problem instance.
    skill_vector
        A "matrix" of size (time_horizon, num_arms) where each row contains a vector of scalars based on the Plackett-Luce model.
        Each scalar is the dot product of the context vector and the maximum-likelihood estimate of the weight parameter for each arm.
    confidence
        A "matrix" of size (time_horizon, num_arms) where each row contains a vector of scalars.
        Each scalar is the confidence bound for each arm of a problem instance.
    gamma
        A hyperparameter of the Polyak-Ruppert averaged SGD method.
    alpha
        A hyperparameter of the Polyak-Ruppert averaged SGD method.
    grad_op_sum
        The quantity to measure V_hat on page 7 in the paper https://arxiv.org/pdf/2002.04275.pdf
    hessian_sum
        The quantity to measure S_hat on page 7 in the paper https://arxiv.org/pdf/2002.04275.pdf
    preference_estimate
        A `PreferenceEstimate` object for entering the samples of the estimates.
    time_step
        The current time step.
    selection
        The subset selection of size `subset_size` from the arms. Start with random selection.
    winner
        The winner arm in the subset selection.
    execution_time
        A variable to track the execution period of the arm strategy algorithm.
    """

    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        omega: Optional[float] = None,
        subset_size: Optional[int] = multiprocessing.cpu_count(),
        parametrizations: Optional[np.array] = None,
        features: Optional[np.array] = None,
        context_matrix: Optional[np.array] = None,
        context_dimensions: Optional[int] = None,
        running_time: Optional[np.array] = None,
        logger_name="BaseAlgorithm",
        logger_level=logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.logger.info("Initializing...")
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        # Extract the features, parameterizations and run times if not already provided based on the solver used.
        if features is None and parametrizations is None and running_time is None:
            if solver == Solver.SAPS.value:
                self.parametrizations = utility_functions.get_parameterization_saps()
                self.features = utility_functions.get_features_saps()
                self.running_time = utility_functions.get_run_times_saps()
            elif solver == Solver.MIPS.value:
                self.parametrizations = utility_functions.get_parameterization_mips()
                self.features = utility_functions.get_features_mips()
                self.running_time = utility_functions.get_run_times_mips()
        else:
            self.parametrizations = parametrizations
            self.features = features
            self.running_time = running_time

        # The number of arms is equal to the number of parameterization sequences.
        self.num_arms = self.parametrizations.shape[0]

        # Initialize a feedback mechanism for the arm staretegy which determines which
        self.feedback_mechanism = MultiDuelFeedback(
            num_arms=self.num_arms, random_state=self.random_state
        )
        self.logger.debug(f"    -> Num arms: {self.num_arms}")
        self.time_horizon = self.features.shape[0]
        self.logger.debug(f"    -> Time Horizon: {self.time_horizon}")
        try:
            assert (
                self.num_arms > subset_size
            ), f"Subset size must be less than than number of arms :{self.num_arms}, given {subset_size}"
        except AssertionError:
            raise
        self.subset_size = subset_size
        self.logger.debug(f"    -> Subset size: {self.subset_size}")

        if context_dimensions is None:
            if joint_featured_map_mode == JointFeatureMode.KRONECKER.value:
                self.context_dimensions = (
                    self.parametrizations.shape[1] * self.features.shape[1]
                )
            elif joint_featured_map_mode == JointFeatureMode.CONCATENATION.value:
                self.context_dimensions = (
                    self.parametrizations.shape[1] + self.features.shape[1]
                )
            elif joint_featured_map_mode == JointFeatureMode.POLYNOMIAL.value:
                self.context_dimensions = 4
                for index in range(
                    (self.features.shape[1] + self.parametrizations.shape[1]) - 2
                ):
                    self.context_dimensions = self.context_dimensions + 3 + index
        else:
            self.context_dimensions = context_dimensions

        if context_matrix is None:
            self.context_matrix = utility_functions.get_context_matrix(
                parametrizations=self.parametrizations,
                features=self.features,
                joint_feature_map_mode=joint_featured_map_mode,
                context_feature_dimensions=self.context_dimensions,
            )
        else:
            self.context_matrix = context_matrix
        self.logger.debug(f"    -> Context matrix shape: {self.context_matrix.shape}")

        # Initialize randomly
        self.theta_init = self.random_state.rand(self.context_dimensions)

        # maximum-likelihood estimate of the weight parameter
        self.theta_hat = copy.copy(self.theta_init)
        self.theta_bar = copy.copy(self.theta_hat)
        self.regret = np.zeros(self.time_horizon)
        if not omega:
            self.omega = 1
        self.skill_vector = np.zeros((self.time_horizon, self.num_arms))
        self.confidence = np.zeros((self.time_horizon, self.num_arms))
        self.gamma = 2
        self.alpha = 0.6
        self.grad_op_sum = np.zeros((self.context_dimensions, self.context_dimensions))
        self.hessian_sum = np.zeros((self.context_dimensions, self.context_dimensions))
        self.preference_estimate = PreferenceEstimate(num_arms=self.num_arms)
        self.time_step = 0
        self.selection = self.random_state.choice(
            self.num_arms, self.subset_size, replace=False
        )
        self.winner = None
        self.execution_time = 0

    def step(self) -> None:
        """Run one step of the algorithm.

        If you want to run an algorithm you should use the ``run`` function
        instead.
        """
        raise NotImplementedError

    def run(self):
        """Run the algorithm until completion.

        Usually, the completion of all the arm strategy algorithms are determined by solving all the problem instances.
        """
        print("Running algorithm...")
        start_time = perf_counter()

        for self.time_step in tqdm(range(1, self.time_horizon + 1)):
            self.step()

        end_time = perf_counter()
        self.execution_time = end_time - start_time
        print("Algorithm Finished...")

    def get_skill_vector(self, theta: np.array, context_vector: np.array) -> np.array:
        """Compute the estimated contextualized utility parameters for each arm in the current time step.

        Parameters
        ----------
        theta : np.array
            The estimated weight parameter.
        context_vector : np.array
            The contextual vector at the current time step.

        Returns
        -------
        skill_vector: np.array
            The estimated contextualized utility parameter for each arm based on the PL-model.
        """
        skill_vector = np.zeros(self.feedback_mechanism.get_num_arms())
        for arm in range(self.feedback_mechanism.get_num_arms()):
            skill_vector[arm] = np.exp(np.inner(theta, context_vector[arm, :]))
        return skill_vector

    # def get_selection_v2(self, quality_of_arms):
    #     best_arm = np.array([(quality_of_arms).argmax()])
    #     contrast_vector = np.empty(
    #         self.feedback_mechanism.get_num_arms(), dtype=np.ndarray
    #     )
    #     for arm in self.feedback_mechanism.get_arms():
    #         contrast_vector[arm] = self.get_contrast_vector(
    #             context_vector_i=self.context_vector[arm, :],
    #             context_vector_j=self.context_vector[best_arm, :],
    #         )
    #     self.contrast_skill_vector[self.time_step - 1] = self.get_contrast_skill_vector(
    #         theta=self.theta_hat, contrast_vector=contrast_vector
    #     )
    #     self.confidence_width_bound[self.time_step - 1] = self.get_confidence_bounds(
    #         selection=self.selection,
    #         time_step=self.time_step,
    #         context_vector=contrast_vector,
    #         winner=self.winner,
    #     )
    #     quality_of_candidates = (
    #         self.contrast_skill_vector[self.time_step - 1]
    #         + self.confidence_width * self.confidence_width_bound[self.time_step - 1]
    #     )
    #     candidates = (-quality_of_candidates).argsort()[0 : self.subset_size]
    #     candidates = np.setdiff1d(candidates, best_arm)
    #     selection = np.append(best_arm, candidates)
    #     return selection

    # def get_contrast_skill_vector(self, theta, contrast_vector):
    #     # compute estimated contextualized utility parameters
    #     contrast_skill_vector = np.zeros(self.feedback_mechanism.get_num_arms())
    #     for arm in range(self.feedback_mechanism.get_num_arms()):
    #         contrast_skill_vector[arm] = np.inner(theta, contrast_vector[arm])
    #     return contrast_skill_vector

    # def get_contrast_vector(self, context_vector_i, context_vector_j):
    #     return (context_vector_i - context_vector_j).reshape(-1)

    def get_confidence_bounds(
        self,
        selection: np.array,
        time_step: int,
        context_vector: np.array,
        winner: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the confidence bounds using the contextual information and the estimated utility weight vector used in some of the algorithms (UCB, Colstim and Colstim_v2).

        Parameters
        ----------
        selection : np.array
            The subset of arms from the pool of arms selected at the current time step.
        time_step : int
            The current time step.
        context_vector : np.array
            The context vector observed in the current time step.
        winner : Optional[int], optional
            The winner arm in the selection of current time step observed by the feedback mechanism, by default None

        Returns
        -------
        np.ndarray
            A numpy array of shape (1, num_arms) containing the confidence bounds of each arm.
        """
        if time_step > 1:
            if winner is not None:
                V_hat = self.compute_V_hat(
                    selection=selection,
                    time_step=time_step,
                    context_vector=context_vector,
                    winner=winner,
                )
            else:
                # If this is the initial time step, the winner is not known, therefore, select the first arm in the selection.
                V_hat = self.compute_V_hat(
                    selection=selection,
                    time_step=time_step,
                    context_vector=context_vector,
                    winner=selection[0],
                )

            # Compute the estimates of the SVD components of the covariance matrix sigma from the paper https://arxiv.org/pdf/2002.04275.pdf
            S_hat = self.compute_S_hat(selection, time_step, context_vector)
            sigma_hat = self.compute_sigma_hat(time_step, V_hat, S_hat)
            gram_matrix = self.compute_gram_matrix_theta_bar(context_vector)
            I_hat = self.compute_I_hat(sigma_hat, gram_matrix)
            I_hat_sqrt = np.sqrt(I_hat)

            # compute confidence bound.
            c_t = (
                self.omega
                * np.sqrt(
                    2 * np.log(time_step)
                    + self.context_dimensions
                    + 2 * np.sqrt(self.context_dimensions * np.log(time_step))
                )
                * I_hat_sqrt
            )
            return c_t
        else:
            return 0

    def compute_V_hat(
        self, selection: np.array, time_step: int, context_vector: np.array, winner: int
    ) -> np.array:
        """Compute the estimated V component of covariance matrix in SVD.

        Parameters
        ----------
        selection : np.array
            The subset of arms from the pool of arms selected at the current time step.
        time_step : int
            The current time step.
        context_vector : np.array
            The context vector observed in the current time step.
        winner : int
            The winner arm in the selection of current time step observed by the feedback mechanism.

        Returns
        -------
        np.array
            The estimated V component of covariance matrix in SVD.
        """
        # compute V_hat using the gradient matrix on the loss function.
        gradient = utility_functions.gradient(
            theta=self.theta_bar,
            winner_arm=winner,
            selection=selection,
            context_vector=context_vector,
        )
        self.grad_op_sum += np.outer(gradient, gradient)
        V_hat = np.asarray((1 / time_step) * self.grad_op_sum).astype("float64")
        return V_hat

    def compute_S_hat(
        self, selection: np.array, time_step: int, context_vector: np.array
    ) -> np.array:
        """Compute the estimated S component of covariance matrix in SVD.

        Parameters
        ----------
        selection : np.array
            The subset of arms from the pool of arms selected at the current time step.
        time_step : int
            The current time step.
        context_vector : np.array
            The context vector observed in the current time step.

        Returns
        -------
        np.array
            The estimated S component of covariance matrix in SVD.
        """
        # compute S_hat using the hessian matrix of the loss function.
        self.hessian_sum += utility_functions.hessian(
            self.theta_bar, selection, context_vector
        )
        S_hat = np.asarray((1 / time_step) * self.hessian_sum).astype("float64")
        return S_hat

    def compute_sigma_hat(
        self, time_step: int, V_hat: np.array, S_hat: np.array
    ) -> np.ndarray:
        """Compute the estimate of the covariance matrix sigma.

        Parameters
        ----------
        time_step : int
            The current time step.
        V_hat : np.array
            The estimated V component of covariance matrix in SVD.
        S_hat : np.ndarray
            The estimated S component of covariance matrix in SVD.

        Returns
        -------
        np.array
            The covariance matrix of the SVD.
        """
        # compute sigma_hat
        try:
            S_hat_inv = np.linalg.inv(S_hat).astype("float64")
        except np.linalg.LinAlgError as error:
            S_hat_inv = np.abs(np.linalg.pinv(S_hat).astype("float64"))
        sigma_hat = (1 / time_step) * np.dot(np.dot(S_hat_inv, V_hat), S_hat_inv)
        sigma_hat = np.nan_to_num(sigma_hat)
        return sigma_hat

    def compute_gram_matrix_theta_bar(self, context_vector: np.array) -> np.ndarray:
        """Compute the gram matrix, Corresponds to `M_t^(i)(theta)` in the paper https://arxiv.org/pdf/2002.04275.pdf

        Parameters
        ----------
        context_vector : np.array
            The context vector observed in the current time step.

        Returns
        -------
        np.ndarray
            The gram matrix.
        """
        # compute gram_matrix of theta_bar
        gram_matrix = np.zeros(
            (self.num_arms, self.context_dimensions, self.context_dimensions)
        )
        for i in range(self.num_arms):
            gram_matrix[i] = np.exp(
                2 * np.dot(context_vector[i], self.theta_bar)
            ) * np.outer(context_vector[i], context_vector[i])
        return gram_matrix

    def compute_I_hat(
        self, sigma_hat: np.ndarray, gram_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate the corresponding I_hat_t in the literature.

        Parameters
        ----------
        sigma_hat : np.ndarray
            The estimated covariance matrix of the SVD.
        gram_matrix : np.ndarray
            The gram matrix.

        Returns
        -------
        np.ndarray
        """
        # compute I_hat
        sigma_hat_sqrt = np.sqrt(sigma_hat)
        I_hat = np.array(
            [
                np.linalg.norm(
                    np.nan_to_num(
                        np.dot(
                            np.nan_to_num(np.dot(sigma_hat_sqrt, gram_matrix[i])),
                            sigma_hat_sqrt,
                        )
                    ),
                    ord=2,
                )
                for i in range(self.num_arms)
            ]
        )
        return I_hat

    def get_selection(self, quality_of_arms: np.array) -> np.array:
        """Get the subset of arms from the pool based on the quality of the arms.

        Parameters
        ----------
        quality_of_arms : np.array
            The quality of each arm. Can be composed of the skill vector along with the confidence bound.

        Returns
        -------
        np.array
            A sorted list of size `subset_size` based on the quality of the arms.
        """
        return (-quality_of_arms).argsort()[0 : self.subset_size]

    def update_estimated_theta(
        self, selection: np.array, time_step: int, winner: int
    ) -> None:
        """Update the estimate of the unknown weight parameter of the contexts.

        Parameters
        ----------
        selection : np.array
            The subset of size `subset_size` from the pool of arms selected in the current time step.
        time_step : int
            The current time step.
        winner : int
            The winner arm in the selected subset.
        """
        context_vector = self.context_matrix[time_step - 1]

        # update step size
        gamma_t = self.gamma * time_step ** ((-1) * self.alpha)
        # update theta_hat with SGD
        self.theta_hat = utility_functions.stochastic_gradient_descent(
            theta=self.theta_hat,
            gamma_t=gamma_t,
            selection=selection,
            context_vector=context_vector,
            winner=winner,
        )

    def update_mean_theta(self, time_step: int) -> None:
        """Update the Polyak-Ruppert average SGD weight parameter.

        Parameters
        ----------
        time_step : int
            The current time step
        """
        # update theta_bar
        self.theta_bar = (
            time_step - 1
        ) * self.theta_bar / time_step + self.theta_hat / time_step

    def get_regret(self) -> np.array:
        """Return the regret computed in all the time steps.

        Returns
        -------
        np.array
            The regret of shape (time_horizon).
        """
        return self.regret

    def compute_regret(self, selection: np.array, time_step: int) -> None:
        """Compute the regret on the selected subset of arms at the current time step.

        Parameters
        ----------
        selection : np.array
            The subset of arms from the pool of arms selected based on the quality of the arms at the current time step.
        time_step : int
            The current time step.
        """
        self.regret[time_step - 1] = metrics.regret_preselection(
            skill_vector=self.running_time[time_step - 1], selection=selection
        )
