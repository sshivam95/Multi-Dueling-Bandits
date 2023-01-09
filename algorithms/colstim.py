"""An implementation of the CoLSTIM Explore-Exploit algorithms as an arm selection strategy."""
import logging
from time import perf_counter
from typing import Optional

import numpy as np
from tqdm import tqdm

from algorithms.algorithm import Algorithm
from util.constants import JointFeatureMode, Solver


class ColstimContextExploreExploit(Algorithm):
    """The ColstimContextExploreExploit class implements the CoLSTIM-Context Explore-Exploit algorithm which using Context matrix to select the subset of arms.
    
    This arm strategy is a modification of the strategy introduced by Bengs et. al. (2022) https://arxiv.org/pdf/2202.04593.pdf for the dueling bandits.
    This class is an extension of that algorithm for the multi-dueling bandits variant of the dueling bandits.
    The algorithm chooses half of the subset of arms by setting a confidence bound on the estimated utility parameter
    and the other half by using the estimated utility parameters.

    Parameters
    ----------
    threshold_parameter: float, optional
        Threshold parameter of CoLSTIM (hyperparameter of CoLSTIM), by default None.
    confidence_width: float, optional
        Confidence width parameter of CoLSTIM (hyperparameter of CoLSTIM), by default None.
        
    Attributes
    ----------
    perturbation_variable: np.array
        Sampled perturbation variable for arm `i` at time step `t` from the Gumbel distribution.
    trimmed_sampled_perturbation_variable: np.array
        Trimmed sampled perturbation variable for arm `i` at time step `t`.
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
        exploration_length: Optional[int] = None,
        threshold_parameter: Optional[float] = None,
        confidence_width: Optional[float] = None,
        logger_name="ColstimContextExploreExploit",
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
        """A single logical step in the CoLSTIM algorithm."""
        # Observe the context
        context_vector = self.context_matrix[self.time_step - 1]

        # Sample the pertubation variable from the Gumbel distribution
        self.sample_perturbation_variable()
        
        # Calculate the skill vector
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            theta=self.theta_bar, context_vector=context_vector
        )
        
        # Calculate the confidence bounds
        self.confidence[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=context_vector,
            winner=self.winner,
        )
        
        # Trim the perturbation variable to match the confidence threshold
        self.update_trimmed_perturbation_variable()
        
        # Select the subset using the Explore-Exploit strategy
        self.selection = self.get_selection_explore_exploit()
        self.logger.debug(f"    -> Selection: {self.selection}")

        self.logger.debug("Starting Duels...")
        # Get the winner feedback
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.logger.debug("Duels finished...")
        self.logger.debug(f"    -> Selection Winner: {self.winner}")
        
        # Update the statistics
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
        """Sample the perturbation variable for arm `i` in the current time step `t` from the Gumbel distribution."""
        for arm in self.feedback_mechanism.get_arms():
            self.perturbation_variable[arm] = self.random_state.gumbel()

    def update_trimmed_perturbation_variable(self):
        """Update the trimmed sampled perturbation variable for arm `i` in the current time step `t`."""
        for arm in self.feedback_mechanism.get_arms():
            self.trimmed_sampled_perturbation_variable[arm] = min(
                self.threshold_parameter,
                max(-(self.threshold_parameter), self.perturbation_variable[arm]),
            )

    def get_selection_explore_exploit(self):
        """Choose the subset by selecting one half from exploitation factor and the other half from the exploration factor.

        Returns
        -------
        np.array
            The selected subset from the pool of arms.
        """
        # Select the exploitation factor
        quality_of_arms_exploit = self.skill_vector[self.time_step - 1]
        selection_exploit = self.get_selection(
            quality_of_arms=quality_of_arms_exploit,
            subset_size=self.subset_size,
        )

        # Select the confidence of only those arms which are not selected in the exploitation section.
        # This deals with the exploration part of selecting the subset.
        mask = np.zeros(self.num_arms, bool)
        mask[selection_exploit] = True
        confidence_temp = (
            self.trimmed_sampled_perturbation_variable
            * self.confidence[self.time_step - 1]
        )
        confidence_temp[mask] = np.nan
        
        # Select the exploration factor
        quality_of_arms_explore = confidence_temp
        selection_explore = self.get_selection(
            quality_of_arms=quality_of_arms_explore,
            subset_size=self.subset_size,
        )

        # If the subset_size is even, choose the arms of equal size.
        # If the subset_size is even, choose the np.floor(subset_size - 1/2) number of arms from the explore part
        # and np.floor(subset_size - 1/2) + 1 arms from the exploit part.
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


class ColstimContrastExploreExploit(ColstimContextExploreExploit):
    """The Colstim class which implements the CoLSTIM algorithm which using Contrast matrix to select the subset of arms using Explore-Exploit method.
    
    This arm strategy is a modification of the strategy introduced by Bengs et. al. (2022) https://arxiv.org/pdf/2202.04593.pdf for the dueling bandits.
    This class is an extension of that algorithm for the multi-dueling bandits variant of the dueling bandits.
    The algorithm chooses half of the subset of arms by setting a confidence bound on the estimated utility parameter
    and the other half by using the estimated utility parameters.
        
    Attributes
    ----------
    contrast_skill_vector: np.array
        Each row corresponds to the skill score of the arms in each time step based on the contrast matrix.
    confidence_width_bound: np.array
        The confidence bounds based on the contrast matrix and the confidence width.
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
        exploration_length: Optional[int] = None,
        threshold_parameter: Optional[float] = None,
        confidence_width: Optional[float] = None,
        logger_name="ColstimContrastExploreExploit",
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

    def step(self):
        # Observe the context vector at the current time step.
        self.context_vector = self.context_matrix[self.time_step - 1]
        
        # If the current time step is not the initial time step then update the estimates.
        if self.time_step > 1:
            self.update_estimated_theta(
                selection=self.selection, time_step=self.time_step, winner=self.winner
            )
            self.update_mean_theta(self.time_step)
            
        # Sample the perturbation variable from Gumbel distribution.
        self.sample_perturbation_variable()
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            theta=self.theta_bar, context_vector=self.context_vector
        )
        
        # Compute the confidence bounds based on the context information.
        self.confidence[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=self.context_vector,
            winner=self.winner,
        )
        
        # Update the perturbation variable.
        self.update_trimmed_perturbation_variable()
        
        # Compute the quality of arms based on the skill vector and coupling of perturbation variable and confidence bounds.
        quality_of_arms = (
            self.skill_vector[self.time_step - 1]
            + self.trimmed_sampled_perturbation_variable
            * self.confidence[self.time_step - 1]
        )
        
        # Get the subset of arms from the contrast matrix.
        self.selection = self.get_selection_contrast(quality_of_arms=quality_of_arms)
        self.logger.debug(f"    -> Selection: {self.selection}")

        self.logger.debug("Starting Duels...")
        # Observe the feedback and get the winner from the subset. 
        # The winner is the one which has the lowest run time to solve the problem instance.
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.logger.debug("Duels finished...")
        self.logger.debug(f"    -> Selection Winner: {self.winner}")
        
        # Update the statistics.
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        self.compute_regret(selection=self.selection, time_step=self.time_step)

    def get_selection_contrast(self, quality_of_arms):
        """Get the subset of arms based on the contrast matrix. 
        
        The best arm is selected using both estimated utility vector (skill_vector) and confidence bound.
        The candidate set of the subset is chosen from the Explore-Exploit strategy.

        Parameters
        ----------
        quality_of_arms : np.array
            The quality of arms is based on the skill vector and the confidence bounds of each arm.

        Returns
        -------
        selection: np.array
            The subset of arms sampled from the pool of arms based on contrast matrix.
        """
        # Select the best arm based on the quality of arms.
        best_arm = np.array([(quality_of_arms).argmax()])
        
        # Create a contrast matrix with respect to the best arm selected.
        contrast_vector = np.zeros((self.num_arms, self.context_dimensions))
        for arm in self.feedback_mechanism.get_arms():
            contrast_vector[arm] = self.get_contrast_vector(
                context_vector_i=self.context_vector[arm, :],
                context_vector_j=self.context_vector[best_arm, :],
            )
            
        # Create a skill vector based on the contrast information.
        self.contrast_skill_vector[self.time_step - 1] = self.get_contrast_skill_vector(
            theta=self.theta_bar, contrast_vector=contrast_vector
        )
        
        # Compute the confidence bounds based on the contrast information instead of the context information.
        self.confidence_width_bound[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=contrast_vector,
            winner=self.winner,
        )
        
        # Get the candidate set.
        candidates = self.get_selection_explore_exploit()
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

    def get_selection_explore_exploit(self):
        """Choose the subset by selecting one half from exploitation factor and the other half from the exploration factor.

        Returns
        -------
        np.array
            The selected subset from the pool of arms.
        """
        # Select the exploitation factor
        quality_of_arms_exploit = self.contrast_skill_vector[self.time_step - 1]
        selection_exploit = self.get_selection(
            quality_of_arms=quality_of_arms_exploit,
            subset_size=self.subset_size,
        )

        # Select the confidence of only those arms which are not selected in the exploitation section.
        # This deals with the exploration part of selecting the subset.
        mask = np.zeros(self.num_arms, bool)
        mask[selection_exploit] = True
        confidence_temp = (
            self.confidence_width * self.confidence_width_bound[self.time_step - 1]
        )
        confidence_temp[mask] = np.nan
        
        # Select the exploration factor
        quality_of_arms_explore = confidence_temp
        selection_explore = self.get_selection(
            quality_of_arms=quality_of_arms_explore,
            subset_size=self.subset_size,
        )

        # If the subset_size is even, choose the arms of equal size.
        # If the subset_size is even, choose the np.floor(subset_size - 1/2) number of arms from the explore part
        # and np.floor(subset_size - 1/2) + 1 arms from the exploit part.
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
