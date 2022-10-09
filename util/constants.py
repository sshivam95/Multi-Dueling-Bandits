"""Constants used in the project."""
from enum import Enum


class Solver(Enum):
    """Constant class for solvers"""

    SAPS = "saps"
    MIPS = "mips"


class JointFeatureMode(Enum):

    POLYNOMIAL = "polynomial"
    CONCATENATION = "concatenation"
    KRONECKER = "kronecker"
