"""Various mechanisms for comparing arms."""

from feedback.feedback_mechanism import FeedbackMechanism
from feedback.multi_duel_feedback import MultiDuelFeedback

feedback_list = [MultiDuelFeedback]
__all__ = [
    "FeedbackMechanism",
    "MultiDuelFeedback",
]
