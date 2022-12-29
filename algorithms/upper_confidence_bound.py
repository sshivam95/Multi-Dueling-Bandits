"""An implementation of the Upper Confidence Bound arm selection strategy algorithm."""
import logging
from typing import Optional

import numpy as np

from algorithms.algorithm import Algorithm
from util.constants import JointFeatureMode, Solver


class UCB(Algorithm):
    """This class implements the Upper Confidence Bound arm selection strategy initially used in the CPPL framework.
    
    The algorithm chooses the subset of arms by setting a confidence bound on the estimated utility parameter.
    The arms with high observed rewards have a high value of estimated utility value and thus a relatively high value in the skill vector and are pulled or selected from the set of arms [n] more often, hence, exploitation. 
    On the other hand, the confidence bounds, get smaller the more often an arm is selected in the subset.
    So an arm that was pulled only rarely has a high value of ct and thus again a high value in the skill vector and is pulled with high probability in the subsequent iterations, hence exploration. 
    After the arm is pulled, the confidence bound shrinks a bit because we get another observation for the pulled arm and are more confident about the estimated utility parameter.
    
    The algorithm implemented in this class is from Mesaoudi et.al. (2020) https://arxiv.org/pdf/2002.04275.pdf.
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

    def step(self) -> None:
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

        # Compute the quality of arms using the skill vector and the confidence bounds
        quality_of_arms = (
            self.skill_vector[self.time_step - 1] + self.confidence[self.time_step - 1]
        )
        self.logger.debug(f"    -> Quality of arms: {quality_of_arms}")

        # Choose the subset of the arms from the pool of arms using the quality of arms
        self.selection = self.get_selection(quality_of_arms=quality_of_arms)
        self.logger.debug(f"    -> Selection: {self.selection}")

        self.logger.debug("Starting Duels...")
        
        # Observe the winner feedback from the feedback mechanism
        self.winner = self.feedback_mechanism.multi_duel(
            selection=self.selection, running_time=self.running_time[self.time_step - 1]
        )
        self.logger.debug("Duels finished...")

        self.logger.debug(f"    -> Selection Winner: {self.winner}")
        
        # Update the samples.
        self.preference_estimate.enter_sample(winner_arm=self.winner)
        
        # Update the estimate statistics
        self.update_estimated_theta(
            selection=self.selection, time_step=self.time_step, winner=self.winner
        )
        self.update_mean_theta(self.time_step)
        
        # Compute the regret
        self.compute_regret(selection=self.selection, time_step=self.time_step)
