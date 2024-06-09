from dataclasses import dataclass
from typing import Any

from cv2.typing import MatLike

from .subtitle import Srt
from ..util.call.anim import call_anim_frame_warp
from ..util.call.resume import call_get_resume_vars


@dataclass(init=True, frozen=False, repr=False, eq=True)
class ImageFrame:
    image: MatLike
    index: int


# TODO freeze..
@dataclass(frozen=False)
class Turbo:
    steps: int
    prev: ImageFrame
    next: ImageFrame

    @staticmethod
    def create(init):
        steps = 1 if init.has_video_input() else init.cadence()
        return Turbo(steps, ImageFrame(None, 0), ImageFrame(None, 0))

    def advance(self, init, i: int, depth):
        if self.is_advance_prev(i):
            self.prev.image, _ = call_anim_frame_warp(init, i, self.prev.image, depth)
        if self.is_advance_next(i):
            self.next.image, _ = call_anim_frame_warp(init, i, self.next.image, depth)

    def progress_step(self, indexes, opencv_image):
        self.prev.image, self.prev.index = self.next.image, self.next.index
        self.next.image, self.next.index = opencv_image, indexes.frame.i
        return self.steps

    def _set_up_step_vars(self, init, turbo):
        # determine last frame and frame to start on
        prev_frame, next_frame, prev_img, next_img = call_get_resume_vars(init, turbo)
        if self.steps > 1:
            self.prev.image, self.prev.index = prev_img, prev_frame
            self.next.image, self.next.index = next_img, next_frame

    def find_start(self, init, turbo) -> int:
        """Maybe resume animation (requires at least two frames - see function)."""
        # set start_frame to next frame
        self._set_up_step_vars(init, turbo)
        return self.next.index + 1 if init.is_resuming_from_timestring() else 0

    def has_steps(self):
        return self.steps > 1

    def _has_prev_image(self):
        return self.prev.image is not None

    def is_advance_prev(self, i: int) -> bool:
        return self._has_prev_image() and i > self.prev.index

    def is_advance_next(self, i: int) -> bool:
        return i > self.next.index

    def is_first_step(self) -> bool:
        return self.steps == 1

    def is_first_step_with_subtitles(self, init) -> bool:
        return self.is_first_step() and Srt.is_subtitle_generation_active(init.args.opts.data)

    def is_emit_in_between_frames(self) -> bool:
        return self.steps > 1
