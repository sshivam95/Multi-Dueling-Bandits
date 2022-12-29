"""A generic way to compare two arms against each other."""
from itertools import combinations

import numpy as np


class FeedbackMechanism:
    """Some means of comparing arms."""

    def __init__(self, num_arms: int) -> None:
        self.num_arms = num_arms

    def multi_duel(self, selection: np.array, running_time: np.array) -> np.array:
        """Perform a multi duel in the selection.

        Parameters
        ----------
        selection
            A selection of arms from the initial set.
        running_time
            The running time of each arm in the time step.

        Returns
        -------
        np.array
            Return the winning arm/s.
        """
        raise NotImplementedError

    def get_arms(self) -> list:
        """Get the pool of arms available."""
        return list(range(self.num_arms))

    def get_num_arms(self) -> int:
        """Get the number of arms."""
        return self.num_arms
