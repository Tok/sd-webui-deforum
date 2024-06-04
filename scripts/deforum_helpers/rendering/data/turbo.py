from dataclasses import dataclass
from typing import Any

from .subtitle import Srt


# TODO freeze..
@dataclass(frozen=False)
class Turbo:
    steps: int
    prev_image: Any
    prev_frame_idx: int
    next_image: Any
    next_frame_idx: int

    @staticmethod
    def create(init):
        steps = 1 if init.has_video_input() else init.cadence()
        return Turbo(steps, None, 0, None, 0)

    def set_up_step_vars(self, prev_img, prev_frame, next_img, next_frame):
        if self.steps > 1:
            self.prev_image, self.prev_frame_idx = prev_img, prev_frame
            self.next_image, self.next_frame_idx = next_img, next_frame

    def _has_prev_image(self):
        return self.prev_image is not None

    def is_advance_prev(self, i) -> bool:
        return self._has_prev_image() and i > self.prev_frame_idx

    def is_advance_next(self, i) -> bool:
        return i > self.next_frame_idx

    def is_first_step(self) -> bool:
        return self.steps == 1

    def is_first_step_with_subtitles(self, init) -> bool:
        return self.is_first_step() and Srt.is_subtitle_generation_active(init.args.opts.data)

    def is_emit_in_between_frames(self) -> bool:
        return self.steps > 1
