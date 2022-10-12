"""Various mechanisms for comparing arms."""

from feedback.feedback_mechanism import FeedbackMechanism
from feedback.matrix_feedback import MatrixFeedback
from feedback.multi_duel_feedback import MultiDuelFeedback
from feedback.dueling_feedback import DuelingFeedback

feedback_list = [MultiDuelFeedback, DuelingFeedback]
__all__ = [
    "FeedbackMechanism",
    "MatrixFeedback",
    "MultiDuelFeedback",
    "DuelingFeedback",
]
