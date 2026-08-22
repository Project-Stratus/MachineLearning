"""Gym wrappers for the balloon environments.

Only one survives Layer 1: the decision-interval wrapper that decouples the
agent's decision rate from the physics timestep.  The rest (``ClipReward``,
``DiscreteActions``, ``ReacherRewardWrapper``, ``RelativePosition``,
``MyPolicy``) were tutorial leftovers written against dict observations, a
continuous action space and Reacher's ``info`` keys — none of which this
project has ever had — and were deleted rather than carried.
"""

from environments.wrappers.decision_interval import DecisionIntervalWrapper

__all__ = ["DecisionIntervalWrapper"]
