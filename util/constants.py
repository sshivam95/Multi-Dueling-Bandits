"""Constants used in the project."""
from enum import Enum
from feedback import feedback_list


class Solver(Enum):
    """Constant class for solvers"""

    SAPS = "saps"
    MIPS = "mips"


class JointFeatureMode(Enum):
    """Constant class for the joint feature map"""
    
    POLYNOMIAL = "polynomial"
    CONCATENATION = "concatenation"
    KRONECKER = "kronecker"
