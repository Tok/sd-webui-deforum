from dataclasses import dataclass

from cv2.typing import MatLike
from .subtitle import Srt
from ..util.call_utils import call_anim_frame_warp, call_get_resume_vars
from ..util.utils import context


@dataclass(init=True, frozen=False, repr=False, eq=False)
class ImageFrame:
    image: MatLike
    frame_index: int


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

    def advance(self, init, indexes, depth):
        with context(indexes.tween_frame) as i:
            if self.is_advance_prev(i):
                self.prev.image, _ = call_anim_frame_warp(init, i, self.prev.image, depth)
            if self.is_advance_next(i):
                self.next.image, _ = call_anim_frame_warp(init, i, self.next.image, depth)

    def progress_step(self, indexes, opencv_image):
        self.prev.image, self.prev.frame_index = self.next.image, self.next.frame_index
        self.next.image, self.next.frame_index = opencv_image, indexes.frame
        return self.steps

    def _set_up_step_vars(self, init, turbo):
        # determine last frame and frame to start on
        prev_frame, next_frame, prev_img, next_img = call_get_resume_vars(init, turbo)
        if self.steps > 1:
            self.prev.image, self.prev.frame_index = prev_img, prev_frame
            self.next.image, self.next.frame_index = next_img, next_frame

    def find_start(self, init, turbo):
        """Maybe resume animation (requires at least two frames - see function)."""
        # set start_frame to next frame
        self._set_up_step_vars(init, turbo)
        return self.next.frame_index + 1 if init.is_resuming_from_timestring() else 0

    def has_steps(self):
        return self.steps > 1

    def _has_prev_image(self):
        return self.prev.image is not None

    def is_advance_prev(self, i) -> bool:
        return self._has_prev_image() and i > self.prev.frame_index

    def is_advance_next(self, i) -> bool:
        return i > self.next.frame_index

    def is_first_step(self) -> bool:
        return self.steps == 1

    def is_first_step_with_subtitles(self, init) -> bool:
        return self.is_first_step() and Srt.is_subtitle_generation_active(init.args.opts.data)

    def is_emit_in_between_frames(self) -> bool:
        return self.steps > 1
