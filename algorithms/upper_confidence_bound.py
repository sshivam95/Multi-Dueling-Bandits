"""An implementation of the Upper Confidence Bound Explore-Exploit arm selection strategy algorithm."""
import logging
from typing import Optional

import numpy as np

from algorithms.algorithm import Algorithm
from util.constants import JointFeatureMode, Solver


class UCBExploreExploit(Algorithm):
    """This class implements the Upper Confidence Bound Explore-Exploit arm selection strategy initially used in the CPPL framework.
    
    The algorithm chooses half of the subset of arms by setting a confidence bound on the estimated utility parameter
    and the other half by using the estimated utility parameters. The arms with high observed rewards have a high value of
    estimated utility value and thus a relatively high value in the skill vector and are pulled or selected from the set of arms [n] more often, hence, exploitation. 
    On the other hand, the confidence bounds, get smaller the more often an arm is selected in the subset.
    So an arm that was pulled only rarely has a high value of ct and thus again a high value in the skill vector and is pulled with high probability in the subsequent iterations, hence exploration. 
    After the arm is pulled, the confidence bound shrinks a bit because we get another observation for the pulled arm and are more confident about the estimated utility parameter.
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
        logger_name="BaseAlgorithm",
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
        logger_name="UpperConfidenceBound",
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
            context_matrix=context_matrix,
            context_dimensions=context_dimensions,
            running_time=running_time,
            logger_level=logger_level,
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

    def step(self):
        # Observe the context vectors of the current time step
        self.logger.debug(f"    -> Time Step: {self.time_step}")
        context_vector = self.context_matrix[self.time_step - 1]
        
        # Compute the estimated contextualized utility parameter
        self.skill_vector[self.time_step - 1] = self.get_skill_vector(
            theta=self.theta_bar, context_vector=context_vector
        )
        self.logger.debug(
            f"    -> Skill Vector: {self.skill_vector[self.time_step - 1]}"
        )

        # Compute the confidence bounds
        self.confidence[self.time_step - 1] = self.get_confidence_bounds(
            selection=self.selection,
            time_step=self.time_step,
            context_vector=context_vector,
            winner=self.winner,
        )
        self.logger.debug(f"    -> Confidence: {self.confidence[self.time_step - 1]}")

        # Choose the subset of the arms from the pool of arms using the quality of arms
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
        self.update_estimated_theta(
            selection=self.selection, time_step=self.time_step, winner=self.winner
        )
        self.update_mean_theta(self.time_step)
        self.compute_regret(selection=self.selection, time_step=self.time_step)

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
        confidence_temp = self.confidence[self.time_step - 1]
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
                    selection_explore[0 : int(np.floor((self.subset_size - 1) / 2))],
                    selection_exploit[0 : int((np.floor((self.subset_size - 1) / 2)) + 1)],
                )
            )

        return selection
