"""A generic superclass which contains the common functions required in the implemented PB-MAB algorithms."""

from typing import Optional
from experiments.environments.plackett_luce_model import PlackettLuceModel
import numpy as np


class Algorithm:
    r"""Parent class of all the implemented PB-MAB algorithms.

    Parameters
    ----------
    feedback_mechanism
        An object that describes the environment.
    time_horizon
        The number of duels the algorithm should perform. If a time horizon is
        given, the algorithm should perform exactly as many duels. May be
        ``None``, in which case the algorithm will execute until an
        algorithm-specific termination condition is reached.

    Attributes
    ----------
    wrapped_feedback
        The ``feedback_mechanism`` parameter with an added decorator. This
        feedback mechanism will raise an exception if a time horizon is given
        and a duel would exceed it. The exception is caught in the ``run``
        function.
    time_horizon
        The number of duels the algorithm should perform. If a time horizon is
        given, the algorithm should perform exactly as many duels. May be
        ``None``, in which case the algorithm will execute until an
        algorithm-specific termination condition is reached.
    """

    def __init__(self, time_horizon: Optional[int]):
        self.random_state = np.random.RandomState()
        self.feedback_mechanism = PlackettLuceModel(
            num_arms=30, random_state=self.random_state
        )
        self.time_horizon = time_horizon

    def step(self) -> None:
        """Run one step of the algorithm.

        This corresponds to a logical step of the algorithm and may perform
        multiple comparisons. What exactly a "logical step" is depends on the
        algorithm.

        The feedback mechanism is wrapped with a ``BudgetedFeedbackMechanism``
        therefore conducting duels within a step may raise an exception if a
        time horizon is specified.

        If you want to run an algorithm you should use the ``run`` function
        instead.

        Raises
        ------
        TimeBudgetExceededException
            If the step has been terminated early due to the time horizon. The
            exception class is local to the object. Different instances raise
            different exceptions. The exception class for this algorithm can be
            accessed through the ``exception_class`` attribute of the wrapped
            feedback mechanism.
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """Determine if the algorithm is finished.

        This may be based on a time horizon or a different algorithm-specific
        termination criterion if time_horizon is ``None``.
        """
        if self.time_horizon is None:
            raise NotImplementedError(
                "No time horizon set and no custom termination condition implemented."
            )
        return self.wrapped_feedback.duels_exhausted()

    def run(self) -> None:
        """Run the algorithm until completion.

        Completion is determined through the ``is_finished`` function. Refer to
        its documentation for more details.
        """
        while not self.is_finished():
            try:
                self.step()
            except self.wrapped_feedback.exception_class:
                # Duel budget exhausted
                return
