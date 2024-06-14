from dataclasses import dataclass


@dataclass(init=True, frozen=True, repr=True, eq=True)
class IndexWithStart:
    start: int = 0
    i: int = 0


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Indexes:
    frame: IndexWithStart = None
    tween: IndexWithStart = None

    @staticmethod
    def create(init, turbo):
        frame_start = turbo.find_start(init, turbo)
        tween_start = 0
        return Indexes(IndexWithStart(frame_start, 0), IndexWithStart(tween_start, 0))

    def update_tween(self, i: int):
        self.tween = IndexWithStart(self.tween.start, i)

    def update_tween_start(self, turbo):
        tween_start = max(self.frame.start, self.frame.i - turbo.steps)
        self.tween = IndexWithStart(tween_start, self.tween.i)

    def update_frame(self, i: int):
        self.frame = IndexWithStart(self.frame.start, i)

    def is_not_first_frame(self):
        return self.frame.i > 0

    def is_first_frame(self):
        return self.frame.i == 0
