"""Various arm selection strategy algorithms to solve Pre-bandit Problem in the CPPL framework."""

from algorithms.algorithm import Algorithm
from algorithms.colstim import Colstim, Colstim_v2
from algorithms.self_sparring import IndependentSelfSparring, IndependentSelfSparringContextual, IndependentSelfSparring_v2
from algorithms.thompson_sampling import ThompsonSampling, ThompsonSamplingContextual, ThompsonSampling_v2
from algorithms.upper_confidence_bound import UCB

regret_minimizing_algorithms = [
    UCB,
    Colstim,
    Colstim_v2,
    ThompsonSampling,
    ThompsonSampling_v2,
    ThompsonSamplingContextual,
    IndependentSelfSparring,
    IndependentSelfSparring_v2,
    IndependentSelfSparringContextual
]

# Generate __all__ for tab-completion etc.
__all__ = ["Algorithm"] + [
    algorithm.__name__ for algorithm in regret_minimizing_algorithms
]
