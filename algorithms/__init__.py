"""Various arm selection strategy algorithms to solve Pre-bandit Problem in the CPPL framework."""

from algorithms.colstim import ColstimContext, ColstimContrast
from algorithms.self_sparring import IndependentSelfSparring, IndependentSelfSparringContextual
from algorithms.thompson_sampling import ThompsonSampling, ThompsonSamplingContextual
from algorithms.upper_confidence_bound import UCB

regret_minimizing_algorithms = [
    UCB,
    ColstimContext,
    ColstimContrast,
    ThompsonSampling,
    ThompsonSamplingContextual,
    IndependentSelfSparring,
    IndependentSelfSparringContextual
]

# Generate __all__ for tab-completion etc.
__all__ = ["Algorithm"] + [
    algorithm.__name__ for algorithm in regret_minimizing_algorithms
]
