"""A generic superclass which contains the common functions required in the implemented PB-MAB algorithms."""

import copy
import logging
import multiprocessing
from time import perf_counter
from typing import Optional

import numpy as np
import scipy as sp
from feedback.feedback_mechanism import FeedbackMechanism
from stats.preference_estimate import PreferenceEstimate
from util import metrics, utility_functions
from util.constants import JointFeatureMode, Solver
from util.exceptions import AlgorithmFinishedException


class Algorithm:
    r"""Parent class of all the implemented PB-MAB algorithms."""

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
        logger_name="BaseAlgorithm",
        logger_level=logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.subset_size = subset_size
        self.logger.debug(f"    -> Subset size: {self.subset_size}")
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.feedback_mechanism: FeedbackMechanism = None

        if features is None  and parametrizations is None  and running_time is None:
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

        self.num_arms = self.parametrizations.shape[0]
        self.logger.debug(f"    -> Num arms: {self.num_arms}")
        self.time_horizon = self.features.shape[0]
        self.logger.debug(f"    -> Time Horizon: {self.time_horizon}")

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

        self.logger.debug(f"    -> Context dimensions: {self.context_dimensions}")
        self.context_matrix = utility_functions.get_context_matrix(
            parametrizations=self.parametrizations,
            features=self.features,
            joint_feature_map_mode=joint_featured_map_mode,
            context_feature_dimensions=self.context_dimensions,
        )
        self.logger.debug(f"    -> Context matrix shape: {self.context_matrix.shape}")
        self.theta_init = self.random_state.rand(
            self.context_dimensions
        )  # Initialize randomly
        self.theta_hat = copy.copy(
            self.theta_init
        )  # maximum-likelihood estimate of the weight parameter
        self.theta_bar = copy.copy(self.theta_hat)
        self.regret = np.zeros(self.time_horizon)
        self.regret_preselection = np.zeros(self.time_horizon)
        if not omega:
            self.omega = (
                1  # np.sqrt(self.context_dimensions * np.log(self.time_horizon)) / 100
            )
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
        )  # start with random selection
        self.winner = None
        self.execution_time = 0

    def step(self) -> None:
        """Run one step of the algorithm.

        This corresponds to a logical step of the algorithm and may perform
        multiple comparisons. What exactly a "logical step" is depends on the
        algorithm.

        The feedback mechanism is wrapped with a ``BudgetedFeedbackMechanism``
        therefore conducting duels within a step may raise an exception if a
        time horizon is specified.

        If you want to run an algorithm you should use the ``run`` function
        instead.

        Raises
        ------
        TimeBudgetExceededException
            If the step has been terminated early due to the time horizon. The
            exception class is local to the object. Different instances raise
            different exceptions. The exception class for this algorithm can be
            accessed through the ``exception_class`` attribute of the wrapped
            feedback mechanism.
        """
        raise NotImplementedError

    def run(self):
        self.logger.info("Running algorithm...")
        start_time = perf_counter()

        for self.time_step in range(1, self.time_horizon + 1):
            self.step()

        end_time = perf_counter()
        self.execution_time = end_time - start_time
        print("Execution time: ", self.execution_time)
        self.logger.info("Algorithm Finished...")

    def get_skill_vector(self, context_vector):
        # compute estimated contextualized utility parameters
        skill_vector = np.zeros(
            self.feedback_mechanism.get_num_arms()
        )  # Line 5 in CPPL algorithm
        for arm in range(self.feedback_mechanism.get_num_arms()):
            skill_vector[arm] = np.exp(np.inner(self.theta_bar, context_vector[arm, :]))
        return skill_vector

    def get_confidence_bounds(
        self, selection, time_step, context_vector, winner: Optional[int] = None
    ):
        """_summary_

        Parameters
        ----------
        selection : _type_
            _description_
        time_step : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if time_step > 1:
            if winner is not None:
                # if winner.shape[0] > 1:
                #     # If there are multiple winners from a duel, select anyone at random
                #     winner = self.random_state.choice(winner, size=1, replace=False)
                # else:
                #     winner = winner
                V_hat = self.compute_V_hat(
                    selection=selection,
                    time_step=time_step,
                    context_vector=context_vector,
                    winner=winner,
                )
            else:
                V_hat = self.compute_V_hat(
                    selection=selection,
                    time_step=time_step,
                    context_vector=context_vector,
                    winner=selection[0],
                )
            try:
                S_hat = self.compute_S_hat(selection, time_step, context_vector)

                sigma_hat = self.compute_sigma_hat(time_step, V_hat, S_hat)

                gram_matrix = self.compute_gram_matrix_theta_bar(context_vector)

                I_hat = self.compute_I_hat(sigma_hat, gram_matrix)
                I_hat_sqrt = np.sqrt(I_hat)

                # compute c_t = confidence bound
                c_t = (
                    self.omega
                    * np.sqrt(
                        2 * np.log(time_step)
                        + self.context_dimensions
                        + 2 * np.sqrt(self.context_dimensions * np.log(time_step))
                    )
                    * I_hat_sqrt
                )
            except:
                return 0
            else:
                return c_t
        else:
            return 0

    def compute_V_hat(self, selection, time_step, context_vector, winner):
        """_summary_

        Parameters
        ----------
        selection : _type_
            _description_
        time_step : _type_
            _description_
        context_vector : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # compute V_hat
        gradient = utility_functions.gradient(
            theta=self.theta_bar,
            winner_arm=winner,
            selection=selection,
            context_vector=context_vector,
        )
        self.grad_op_sum += np.outer(gradient, gradient)
        V_hat = np.asarray((1 / time_step) * self.grad_op_sum).astype("float64")
        return V_hat

    def compute_S_hat(self, selection, time_step, context_vector):
        """_summary_

        Parameters
        ----------
        selection : _type_
            _description_
        time_step : _type_
            _description_
        context_vector : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # compute S_hat
        self.hessian_sum += utility_functions.hessian(
            self.theta_bar, selection, context_vector
        )
        S_hat = np.asarray((1 / time_step) * self.hessian_sum).astype("float64")
        return S_hat

    def compute_sigma_hat(self, time_step, V_hat, S_hat):
        """_summary_

        Parameters
        ----------
        time_step : _type_
            _description_
        V_hat : _type_
            _description_
        S_hat : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # compute sigma_hat
        S_hat_inv = np.linalg.inv(S_hat).astype("float64")
        sigma_hat = (1 / time_step) * np.dot(np.dot(S_hat_inv, V_hat), S_hat_inv)
        sigma_hat = np.nan_to_num(sigma_hat)
        return sigma_hat

    def compute_gram_matrix_theta_bar(self, context_vector):
        """_summary_

        Parameters
        ----------
        context_vector : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # compute gram_matrix of theta_bar
        gram_matrix = np.zeros((self.num_arms, self.context_dimensions, self.context_dimensions))
        for i in range(self.num_arms):
            gram_matrix[i] = np.exp(2 * np.dot(context_vector[i], self.theta_bar)) * np.outer(
                context_vector[i], context_vector[i]
            )
        return gram_matrix

    def compute_I_hat(self, sigma_hat, gram_matrix):
        """_summary_

        Parameters
        ----------
        sigma_hat : _type_
            _description_
        gram_matrix : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # compute I_hat
        sigma_hat_sqrt = sp.linalg.sqrtm(sigma_hat)
        I_hat = np.array(
            [
                np.linalg.norm(
                    np.dot(np.dot(sigma_hat_sqrt, gram_matrix[i]), sigma_hat_sqrt), ord=2
                )
                for i in range(self.n_arms)
            ]
        )
        return I_hat

    def get_selection(self, quality_of_arms):
        """_summary_

        Parameters
        ----------
        selection : _type_
            _description_
        time_step : _type_
            _description_
        """
        return (-quality_of_arms).argsort()[0 : self.subset_size]

    def update_theta(self, selection, time_step, winner):
        """_summary_

        Parameters
        ----------
        selection : _type_
            _description_
        time_step : _type_
            _description_
        """
        # if winner.shape[0] > 1:
        #     # If there are multiple winners from a duel, select anyone at random
        #     winner = self.random_state.choice(winner, size=1, replace=False)
        # else:
        #     winner = winner
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
        # update theta_bar
        self.theta_bar = (
            time_step - 1
        ) * self.theta_bar / time_step + self.theta_hat / time_step

    def get_regret(self):
        return self.regret

    def compute_regret(self, selection, time_step):
        self.regret[time_step - 1] = metrics.regret_preselection_saps(
            skill_vector=self.running_time[time_step - 1], selection=selection
        )
        # self.regret_preselection[time_step - 1] = metrics.regret_preselection(
        #     theta=
        # )
