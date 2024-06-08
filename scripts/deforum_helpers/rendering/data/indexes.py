from dataclasses import dataclass


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Indexes:
    frame: int = 0
    start_frame: int = 0
    tween_frame: int = 0
    tween_frame_start: int = 0

    @staticmethod
    def create(init, turbo):
        # TODO try to init everything right away...
        return Indexes(0, turbo.find_start(init, turbo), 0, 0)
