from dataclasses import dataclass
from typing import Any


@dataclass(init=True, frozen=False, repr=False, eq=False)
class TweenStep:
    """cadence vars"""
    tween: float
    cadence_flow: Any
    cadence_flow_inc: Any

    @staticmethod
    def create(init, indexes):
        tween = float(indexes.tween.i - indexes.tween.start + 1) / float(indexes.frame.i - indexes.tween.start)
        return TweenStep(tween, None, None)
