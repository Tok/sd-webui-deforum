from dataclasses import dataclass


@dataclass(init=True, frozen=False, repr=True, eq=True)
class IndexWithStart:
    start: int = 0
    i: int = 0


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Indexes:
    frame: IndexWithStart = None
    tween: IndexWithStart = None

    @staticmethod
    def create(init, turbo):
        frame = IndexWithStart(turbo.find_start(init, turbo), 0)
        tween = IndexWithStart(0, 0)
        return Indexes(frame, tween)
