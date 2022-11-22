import logging
from time import perf_counter
from typing import Optional

import numpy as np
from tqdm import tqdm

from algorithms.algorithm import Algorithm
from feedback.multi_duel_feedback import MultiDuelFeedback
from util.constants import JointFeatureMode, Solver


class Colstim(Algorithm):
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
        exploration_length: Optional[int] = None,
        threshold_parameter: Optional[float] = None,
        confidence_width: Optional[float] = None,
        logger_name="COLSTIM",
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
            context_matrix=context_matrix,
            context_dimensions=context_dimensions,
            logger_level=logger_level,
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.logger.info("Initializing...")

        if exploration_length is not None:
            try:
                assert (
                    exploration_length > 0
                ), f"exploration length must be greater than 0, given {exploration_length}"
            except AssertionError:
                raise
            self.exploration_length = exploration_length
        else:
            self.exploration_length = self.context_dimensions * self.num_arms
        if threshold_parameter is not None:
            try:
                assert (
                    threshold_parameter > 0
                ), f"threshold parameter must be greater than 0, given {threshold_parameter}"
            except AssertionError:
                raise
            self.threshold_parameter = threshold_parameter
        else:
            self.threshold_parameter = np.sqrt(
                self.context_dimensions * np.log(self.time_horizon)
            )
        if confidence_width is not None:
            try:
                assert (
                    confidence_width > 0
                ), f"confidence width must be greater than 0, given {confidence_width}"
            except AssertionError:
                raise
            self.confidence_width = confidence_width
        else:
            self.confidence_width = np.sqrt(
                self.context_dimensions * np.log(self.time_horizon)
            )
        self.perturbation_variable = np.zeros(self.num_arms)
        self.trimmed_sampled_perturbation_variable = np.zeros(self.num_arms)

    def step(self):
        context_vector = self.context_matrix[self.time_step - 1]

        self.sample_perturbation_variable()
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            theta=self.theta_hat, context_vector=context_vector
        )
        self.confidence[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=context_vector,
            winner=self.winner,
        )
        self.update_trimmed_perturbation_variable()
        # quality_of_arms = (
        #     self.skill_vector[self.time_step - 1]
        #     + self.trimmed_sampled_perturbation_variable
        #     * self.confidence[self.time_step - 1]
        # )
        # self.selection = self.get_selection(quality_of_arms=quality_of_arms)
        self.selection = self.get_selection_framework_v2()
        self.logger.debug(f"    -> Selection: {self.selection}")

        self.logger.debug("Starting Duels...")
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.logger.debug("Duels finished...")
        self.logger.debug(f"    -> Selection Winner: {self.winner}")
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        self.logger.debug("Updating Theta...")
        self.update_estimated_theta(
            selection=self.selection,
            time_step=self.time_step,
            winner=self.winner,
        )
        self.update_mean_theta(self.time_step)
        self.compute_regret(selection=self.selection, time_step=self.time_step)

    def sample_perturbation_variable(self):
        for arm in self.feedback_mechanism.get_arms():
            self.perturbation_variable[arm] = self.random_state.gumbel()

    def update_trimmed_perturbation_variable(self):
        for arm in self.feedback_mechanism.get_arms():
            self.trimmed_sampled_perturbation_variable[arm] = min(
                self.threshold_parameter,
                max(-(self.threshold_parameter), self.perturbation_variable[arm]),
            )

    def get_selection_framework_v2(self):
        quality_of_arms_explore = self.skill_vector[self.time_step - 1]
        selection_explore = self.get_selection(
            quality_of_arms=quality_of_arms_explore,
            subset_size=self.subset_size,
        )

        mask = np.zeros(self.num_arms, bool)
        mask[selection_explore] = True
        confidence_temp = (
            self.trimmed_sampled_perturbation_variable
            * self.confidence[self.time_step - 1]
        )
        confidence_temp[mask] = np.nan
        quality_of_arms_exploit = confidence_temp
        selection_exploit = self.get_selection(
            quality_of_arms=quality_of_arms_exploit,
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


class Colstim_v2(Colstim):
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
        exploration_length: Optional[int] = None,
        threshold_parameter: Optional[float] = None,
        confidence_width: Optional[float] = None,
        logger_name="COLSTIM_v2",
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
            context_matrix=context_matrix,
            context_dimensions=context_dimensions,
            exploration_length=exploration_length,
            threshold_parameter=threshold_parameter,
            confidence_width=confidence_width,
            logger_name=logger_name,
            logger_level=logger_level,
        )
        self.contrast_skill_vector = np.zeros((self.time_horizon, self.num_arms))
        self.confidence_width_bound = np.zeros((self.time_horizon, self.num_arms))

    def run(self):
        print("Running algorithm...")
        start_time = perf_counter()

        for self.time_step in tqdm(range(1, self.time_horizon + 1)):
            self.step()

        end_time = perf_counter()
        self.execution_time = end_time - start_time
        print("Algorithm Finished...")

    def step(self):
        self.context_vector = self.context_matrix[self.time_step - 1]
        if self.time_step > 1:
            self.update_estimated_theta(
                selection=self.selection, time_step=self.time_step, winner=self.winner
            )
            self.update_mean_theta(self.time_step)
        self.sample_perturbation_variable()
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            theta=self.theta_hat, context_vector=self.context_vector
        )
        self.confidence[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=self.context_vector,
            winner=self.winner,
        )
        self.update_trimmed_perturbation_variable()
        quality_of_arms = (
            self.skill_vector[self.time_step - 1]
            + self.trimmed_sampled_perturbation_variable
            * self.confidence[self.time_step - 1]
        )
        self.selection = self.get_selection_v2(quality_of_arms=quality_of_arms)
        self.logger.debug(f"    -> Selection: {self.selection}")

        self.logger.debug("Starting Duels...")
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.logger.debug("Duels finished...")
        self.logger.debug(f"    -> Selection Winner: {self.winner}")
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        self.compute_regret(selection=self.selection, time_step=self.time_step)
# TODO: add Framework v2

    def get_selection_v2(self, quality_of_arms):
        best_arm = np.array([(quality_of_arms).argmax()])
        contrast_vector = np.empty(
            self.feedback_mechanism.get_num_arms(), dtype=np.ndarray
        )
        for arm in self.feedback_mechanism.get_arms():
            contrast_vector[arm] = self.get_contrast_vector(
                context_vector_i=self.context_vector[arm, :],
                context_vector_j=self.context_vector[best_arm, :],
            )
        self.contrast_skill_vector[self.time_step - 1] = self.get_contrast_skill_vector(
            theta=self.theta_hat, contrast_vector=contrast_vector
        )
        self.confidence_width_bound[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=contrast_vector,
            winner=self.winner,
        )
        # quality_of_candidates = (
        #     self.contrast_skill_vector[self.time_step - 1]
        #     + self.confidence_width * self.confidence_width_bound[self.time_step - 1]
        # )
        candidates = self.get_selection_framework_v2()
        candidates = np.setdiff1d(candidates, best_arm)
        selection = np.append(best_arm, candidates)
        return selection

    def get_contrast_skill_vector(self, theta, contrast_vector):
        # compute estimated contextualized utility parameters
        contrast_skill_vector = np.zeros(self.feedback_mechanism.get_num_arms())
        for arm in range(self.feedback_mechanism.get_num_arms()):
            contrast_skill_vector[arm] = np.inner(theta, contrast_vector[arm])
        return contrast_skill_vector

    def get_contrast_vector(self, context_vector_i, context_vector_j):
        return (context_vector_i - context_vector_j).reshape(-1)

    def get_selection_framework_v2(self):
        quality_of_arms_explore = self.contrast_skill_vector[self.time_step - 1]
        selection_explore = self.get_selection(
            quality_of_arms=quality_of_arms_explore,
            subset_size=self.subset_size,
        )

        mask = np.zeros(self.num_arms, bool)
        mask[selection_explore] = True
        confidence_temp = (
            self.confidence_width * self.confidence_width_bound[self.time_step - 1]
        )
        confidence_temp[mask] = np.nan
        quality_of_arms_exploit = confidence_temp
        selection_exploit = self.get_selection(
            quality_of_arms=quality_of_arms_exploit,
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
