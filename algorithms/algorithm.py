"""A generic superclass which contains the common functions required in the implemented PB-MAB algorithms."""

import copy
from typing import Optional

import numpy as np
import scipy as sp
from util import utility_functions
from util.constants import JointFeatureMode, Solver
from util.exceptions import AlgorithmFinishedException


class Algorithm:
    r"""Parent class of all the implemented PB-MAB algorithms.

    Attributes
    ----------
    wrapped_feedback
        The ``feedback_mechanism`` parameter with an added decorator. This
        feedback mechanism will raise an exception if a time horizon is given
        and a duel would exceed it. The exception is caught in the ``run``
        function.
    time_horizon
        The number of duels the algorithm should perform. If a time horizon is
        given, the algorithm should perform exactly as many duels. May be
        ``None``, in which case the algorithm will execute until an
        algorithm-specific termination condition is reached.
    """

    def __init__(
        self,
        joint_featured_map_mode: Optional[str] = JointFeatureMode.POLYNOMIAL.value,
        solver: Optional[str] = Solver.SAPS.value,
        omega: Optional[float] = None,
    ) -> None:
        self.random_state = np.random.RandomState()
        if solver == Solver.SAPS.value:
            self.parametrizations = utility_functions.get_parameterization_saps()
            self.features, self.problem_instances = utility_functions.get_features_saps()
            self.running_time = utility_functions.get_run_times_saps()
        self.num_arms = self.parametrizations.shape[0]
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
                self.context_dimensions = (
                    self.context_dimensions + 3 + index
                )
        self.context_matrix = utility_functions.get_context_matrix(
            parametrizations=self.parametrizations,
            features=self.features,
            joint_feature_map_mode=joint_featured_map_mode,
            context_feature_dimensions=self.context_dimensions,
        )
        self.arms = list(range(self.num_arms))
        self.time_horizon = self.features.shape[0]
        self.theta_init = self.random_state.rand(self.context_dimensions)
        self.theta_hat = copy.copy(self.theta_init).reshape(-1,1)
        self.theta_bar = copy.copy(self.theta_hat).reshape(-1,1)
        self.regret = np.zeros(self.time_horizon)
        if not omega:
            self.omega = (
                np.sqrt(self.context_dimensions * np.log(self.time_horizon)) / 100
            )
        self.confidence = np.zeros((self.num_arms, self.num_arms))
        self.gamma = 1
        self.alpha = 0.2
        self.grad_op_sum = np.zeros((self.context_dimensions, self.context_dimensions))
        self.hessian_sum = np.zeros((self.context_dimensions, self.context_dimensions))

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

    def is_finished(self) -> bool:
        """Determine if the algorithm is finished.

        This may be based on a time horizon or a different algorithm-specific
        termination criterion if time_horizon is ``None``.
        """
        raise NotImplementedError(
            "No time horizon set and no custom termination condition implemented."
        )

    def run(self) -> None:
        """Run the algorithm until completion.

        Completion is determined through the ``is_finished`` function. Refer to
        its documentation for more details.
        """
        while not self.is_finished():
            try:
                self.step()
            except AlgorithmFinishedException:
                # Duel budget exhausted
                return

    def get_arms(self) -> list:
        """Get the pool of arms available."""
        return self.arms.copy()

    def get_num_arms(self) -> int:
        """Get the number of arms."""
        return self.num_arms

    def get_confidence_bounds(self, ranking, time_step):
        context_vector = self.context_matrix[time_step - 1]

        V_hat = self.compute_V_hat(ranking, time_step, context_vector)

        S_hat = self.compute_S_hat(ranking, time_step, context_vector)

        sigma_hat = self.compute_sigma_hat(time_step, V_hat, S_hat)

        M = self.compute_M_theta_bar(context_vector)

        I_hat = self.compute_I_hat(sigma_hat, M)
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
        return c_t

    def compute_V_hat(self, ranking, time_step, context_vector):
        # compute V_hat
        sum = 0
        for index, _ in enumerate(ranking):
            sum += utility_functions.gradient(
                theta=self.theta_bar,
                winner_arm=ranking[index],
                subset_arms=ranking[index:],
                context_matrix=context_vector,
            )
        gradient = sum
        self.grad_op_sum += np.outer(gradient, gradient)
        V_hat = np.asarray((1 / time_step) * self.grad_op_sum).astype("float64")
        return V_hat
    
    def compute_S_hat(self, ranking, time_step, context_vector):
        # compute S_hat
        self.hessian_sum += utility_functions.hessian(
            self.theta_bar, ranking, context_vector
        )
        S_hat = np.asarray((1 / time_step) * self.hessian_sum).astype("float64")
        return S_hat
    
    def compute_sigma_hat(self, time_step, V_hat, S_hat):
        # compute sigma_hat
        S_hat_inv = np.linalg.inv(S_hat).astype("float64")
        sigma_hat = (1 / time_step) * np.dot(np.dot(S_hat_inv, V_hat), S_hat_inv)
        sigma_hat = np.nan_to_num(sigma_hat)
        return sigma_hat
    
    def compute_M_theta_bar(self, context_vector):
        # compute M of theta_bar
        M = np.zeros((self.num_arms, self.context_dimensions, self.context_dimensions))
        for i in range(self.num_arms):
            M[i] = np.exp(2 * np.dot(context_vector[i], self.theta_bar)) * np.outer(
                context_vector[i], context_vector[i]
            )
        return M

    def compute_I_hat(self, sigma_hat, M):
        # compute I_hat
        sigma_hat_sqrt = sp.linalg.sqrtm(sigma_hat)
        I_hat = np.array(
            [
                np.linalg.norm(
                    np.dot(np.dot(sigma_hat_sqrt, M[i]), sigma_hat_sqrt), ord=2
                )
                for i in range(self.n_arms)
            ]
        )
        return I_hat

    def get_ranking(self, selection, time_step):
        pass
