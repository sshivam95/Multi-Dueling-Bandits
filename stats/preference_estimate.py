"""Utilities for estimating preferences based on samples."""

from typing import Union

import numpy as np



class PreferenceEstimate:
    """An estimation of the preferences based on samples.

    Parameters
    ----------
    num_arms
        The total number of arms.
    """

    def __init__(
        self,
        num_arms: int,
    ) -> None:
        self.num_arms = num_arms
        self.wins = np.zeros(num_arms)
        self.losses = np.zeros(num_arms)
        self.samples = np.zeros(num_arms)

    def enter_sample(self, winner_arm: Union[int, np.array]) -> None:
        self.wins[winner_arm] += 1
        for arm in range(self.num_arms):
            if arm != winner_arm:
                self.losses[arm] += 1
            self.samples[arm] = self.wins[arm] + self.losses[arm]