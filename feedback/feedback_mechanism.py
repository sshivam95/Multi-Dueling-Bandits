"""A generic way to compare two arms against each other."""
from itertools import combinations
import numpy as np


class FeedbackMechanism:
    """Some means of comparing two arms."""

    def __init__(self, num_arms: int) -> None:
        self.num_arms = num_arms

    # In our final design we will probably want a better arm representation to
    # avoid restricting it to int.
    def duel(self, arm_i_index: int, arm_j_index: int, true_skills: np.array) -> bool:
        """Perform a duel between two arms.

        Parameters
        ----------
        arm_i_index
            The index of challenger arm.
        arm_j_index
            The index of arm to compare against.

        Returns
        -------
        bool
            True if the first arm wins.
        """
        raise NotImplementedError

    def duel_repeatedly(
        self,
        arm_i_index: int,
        arm_j_index: int,
        duel_count: int,
    ) -> int:
        """Perform multiple duels between two arms in a single step.

        Parameters
        ----------
        arm_i_index
            The arm of challenger arm.
        arm_j_index
            The index of arm to compare against.
        duel_count
            The number of rounds ``arm_i_index`` is compared against ``arm_j_index``.
        duel_limit
            The number of duels that the algorithm has budget to allow.

        Returns
        -------
        int
           The number of wins of the first arm against the second arm.
        """
        wins = 0
        for _ in range(duel_count):
            if self.duel(arm_i_index, arm_j_index):
                wins += 1
        return wins

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

    def get_dueling_pair_combinations(self) -> list:
        """Get the possible dueling pair combinations from the participating arms.

        Returns
        -------
        list
            The list of dueling pair combinations.
        """
        return list(combinations(range(self.get_num_arms()), 2))
