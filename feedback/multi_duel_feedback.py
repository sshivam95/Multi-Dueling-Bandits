import numpy as np
from util.utility_functions import argmin_set

from feedback import FeedbackMechanism


class MultiDuelFeedback(FeedbackMechanism):
    def __init__(self, num_arms: int, random_state) -> None:
        super().__init__(num_arms)
        self.random_state = random_state

    def multi_duel(self, selection: np.array, running_time: np.array) -> np.array:
        true_skills = running_time[selection]
        
        # Get the parameterizations which have the least run time.
        # If there are more than one parameterizations like that then break ties randomly.
        winners = argmin_set(true_skills)
        if len(winners) > 1:
            winners = self.random_state.choice(winners)
        return selection[winners]
