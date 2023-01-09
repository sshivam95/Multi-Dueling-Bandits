"""Various algorithms to solve Preference-Based Multi-Armed Bandit Problems."""

from algorithms.algorithm import Algorithm
from algorithms.colstim import ColstimContextExploreExploit, ColstimContrastExploreExploit
from algorithms.upper_confidence_bound import UCBExploreExploit

# Pylint insists that regret_minimizing_algorithms and interfaces are constants and should be
# named in UPPER_CASE. Technically that is correct, but it doesn't feel quite
# right for this use case. Its not a typical constant. A similar use-case would
# be numpy's np.core.numerictypes.allTypes, which is also not names in
# UPPER_CASE.
# pylint: disable=invalid-name

# Make the actual algorithm classes available for easy enumeration in
# experiments and tests.
# All algorithms that include some sort of regret-minimizing mode. That
# includes PAC algorithms with an (optional) exploitation phase.
regret_minimizing_algorithms = [
    UCBExploreExploit,
    ColstimContextExploreExploit,
    ColstimContrastExploreExploit,
]
# This is not really needed, but otherwise zimports doesn't understand the
# __all__ construct and complains that the Algorithm import is unnecessary.
interfaces = [Algorithm]

# Generate __all__ for tab-completion etc.
__all__ = ["Algorithm"] + [
    algorithm.__name__ for algorithm in regret_minimizing_algorithms
]
