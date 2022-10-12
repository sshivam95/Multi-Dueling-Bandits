import numpy as np
from util.utility_functions import argmax_set, argmin_set

from feedback import FeedbackMechanism


class MultiDuelFeedback(FeedbackMechanism):
    def __init__(self, num_arms: int) -> None:
        super().__init__(num_arms)

    def multi_duel(self, selection: np.array, running_time: np.array) -> np.array:
        true_skills = running_time[selection]
        winners = np.argmax(true_skills)
        return winners
