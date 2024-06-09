from dataclasses import dataclass
from typing import Any


@dataclass(init=True, frozen=True, repr=False, eq=False)
class SubStep:
    tween: Any  # cadence vars

    @staticmethod
    def create(init, indexes):
        tween = (float(indexes.tween.i - indexes.tween.start + 1) / float(indexes.frame.i - indexes.tween.start))
        return SubStep(tween)
