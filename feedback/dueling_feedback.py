import numpy as np

from feedback import FeedbackMechanism


class DuelingFeedback(FeedbackMechanism):
    def __init__(self, num_arms: int) -> None:
        super().__init__(num_arms)

    def duel(self, arm_i_index: int, arm_j_index: int, true_skills: np.array) -> bool:
        skills_i = true_skills[arm_i_index]
        skills_j = true_skills[arm_j_index]
        if skills_i > skills_j:
            return True
        else:
            return False     
