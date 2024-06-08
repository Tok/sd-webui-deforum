from dataclasses import dataclass
from typing import Any

from .subtitle import Srt
from ...resume import get_resume_vars


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

    def _set_up_step_vars(self, init, turbo):
        # determine last frame and frame to start on
        prev_frame, next_frame, prev_img, next_img = get_resume_vars(
            folder=init.args.args.outdir,
            timestring=init.args.anim_args.resume_timestring,
            cadence=turbo.steps)
        if self.steps > 1:
            self.prev_image, self.prev_frame_idx = prev_img, prev_frame
            self.next_image, self.next_frame_idx = next_img, next_frame
        return next_frame

    def find_start(self, init, turbo):
        """Maybe resume animation (requires at least two frames - see function)."""
        # set start_frame to next frame
        return self._set_up_step_vars(init, turbo) + 1 if init.is_resuming_from_timestring() else 0

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
