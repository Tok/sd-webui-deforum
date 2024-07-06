from dataclasses import dataclass

from cv2.typing import MatLike

from .subtitle import Srt
from ..util import log_utils
from ..util.call.anim import call_anim_frame_warp
from ..util.call.hybrid import call_get_flow_from_images
from ..util.call.resume import call_get_resume_vars
from ...hybrid_video import (image_transform_ransac, image_transform_optical_flow,
                             abs_flow_to_rel_flow, rel_flow_to_abs_flow)


@dataclass(init=True, frozen=False, repr=False, eq=True)
class ImageFrame:
    image: MatLike | None
    index: int


# TODO freeze..
@dataclass(frozen=False)
class Turbo:
    steps: int  # cadence
    prev: ImageFrame
    next: ImageFrame
    # depth: None

    @staticmethod
    def create(data):
        steps = 1 if data.has_video_input() else data.cadence()
        return Turbo(steps, ImageFrame(None, 0), ImageFrame(None, 0))

    def advance(self, data, i: int, depth):
        if self.is_advance_prev(i) and self.prev.image is not None:
            self.prev.image, _ = call_anim_frame_warp(data, i, self.prev.image, depth)
        if self.is_advance_next(i) and self.next.image is not None:
            self.next.image, _ = call_anim_frame_warp(data, i, self.next.image, depth)

    def do_hybrid_video_motion(self, data, indexes, reference_images):
        """Warps the previous and/or the next to match the motion of the provided reference images."""
        motion = data.args.anim_args.hybrid_motion

        def _is_do_motion(motions):
            return indexes.tween.i > 0 and motion in motions

        if _is_do_motion(['Affine', 'Perspective']):
            Turbo.advance_hybrid_motion_ransac_transform(data, indexes, reference_images)
        if _is_do_motion(['Optical Flow']):
            self.advance_hybrid_motion_optical_tween_flow(data, indexes, reference_images, step)

    def advance_optical_flow(self, tween_step, flow_factor: int = 1):
        flow = tween_step.cadence_flow * -1
        self.next.image = image_transform_optical_flow(self.next.image, flow, flow_factor)

    def advance_optical_tween_flow(self, step, flow):
        ff = step.step_data.flow_factor()
        i = indexes.tween.i
        if self.is_advance_prev(i):
            self.prev.image = image_transform_optical_flow(self.prev.image, flow, ff)
        if self.is_advance_next(i):
            self.next.image = image_transform_optical_flow(self.next.image, flow, ff)

    def advance_hybrid_motion_optical_tween_flow(self, data, indexes, reference_images, step):
        if data.args.anim_args.hybrid_motion_use_prev_img:
            flow = call_get_flow_for_hybrid_motion_prev(data, indexes.tween.i - 1, reference_images.previous)
            turbo.advance_optical_tween_flow(self, step, flow)
            data.animation_mode.prev_flow = flow
        else:
            flow = call_get_flow_for_hybrid_motion(data, indexes.tween.i - 1)
            turbo.advance_optical_tween_flow(self, step, flow)
            data.animation_mode.prev_flow = flow

    def advance_cadence_flow(self, data, tween_step):
        ff = data.args.anim_args.cadence_flow_factor_schedule
        i = tween_step.i()
        inc = tween_step.cadence_flow_inc  # FIXME
        if self.is_advance_prev(i):
            self.prev.image = image_transform_optical_flow(self.prev.image, inc, ff)
        if self.is_advance_next(i):
            self.next.image = image_transform_optical_flow(self.next.image, inc, ff)

    # TODO? move to RenderData
    def advance_ransac_transform(self, data, matrix):
        i = indexes.tween.i
        motion = data.args.anim_args.hybrid_motion
        if self.is_advance_prev(i):
            self.prev.image = image_transform_ransac(self.prev.image, matrix, motion)
        if self.is_advance_next(i):
            self.next.image = image_transform_ransac(self.next.image, matrix, motion)

    # TODO? move to RenderData
    @staticmethod
    def advance_hybrid_motion_ransac_transform(data, indexes, reference_images):
        if data.args.anim_args.hybrid_motion_use_prev_img:
            matrix = call_get_matrix_for_hybrid_motion_prev(data, indexes.tween.i - 1, reference_images.previous)
            turbo.advance_ransac_transform(data, matrix)
        else:
            matrix = call_get_matrix_for_hybrid_motion(data, indexes.tween.i - 1)
            turbo.advance_ransac_transform(data, matrix)

    def advance_optical_flow_cadence_before_animation_warping(self, data, last_step, tween_step):
        if data.is_3d_or_2d() and data.has_optical_flow_cadence():
            i = data.indexes.tween.start
            has_tween_schedule = data.animation_keys.deform_keys.strength_schedule_series[i] > 0
            has_images = self.prev.image is not None and self.next.image is not None
            has_step_and_images = tween_step.cadence_flow is None and has_images
            if has_tween_schedule and has_step_and_images and data.animation_mode.is_raft_active():
                cadence = "RAFT"  # FIXME data.args.anim_args.optical_flow_cadence
                flow = call_get_flow_from_images(data, self.prev.image, self.next.image, cadence)
                tween_step.cadence_flow = (flow / 2)
            if tween_step.cadence_flow is not None:
                self.advance_optical_flow(tween_step)

            flow_factor = 100.0 / len(last_step.tweens)
            if tween_step.cadence_flow is not None:
                self.next.image = image_transform_optical_flow(self.next.image, -tween_step.cadence_flow, flow_factor)

    def do_optical_flow_cadence_after_animation_warping(self, data, indexes, tween_step):
        if not data.animation_mode.is_raft_active():
            return self.next.image
        if tween_step.cadence_flow is not None:
            log_utils.debug("do_optical_flow_cadence_after_animation_warping")
            # TODO Calculate all increments before running the generation (and try to avoid abs->rel->abc conversions).
            temp_flow = abs_flow_to_rel_flow(tween_step.cadence_flow, data.width(), data.height())
            new_flow, _ = call_anim_frame_warp(data, indexes.tween.i, temp_flow, self.depth)
            tween_step.cadence_flow = new_flow
            abs_flow = rel_flow_to_abs_flow(tween_step.cadence_flow, data.width(), data.height())
            tween_step.cadence_flow_inc = abs_flow * tween_step.tween
            self.advance_cadence_flow(data, tween_step)
        self.prev.index = self.next.frame_idx = indexes.tween.i if indexes is not None else 0
        if self.prev.image is not None and tween_step.tween < 1.0:
            return self.prev.image * (1.0 - tween_step.tween) + self.next.image * tween_step.tween
        else:
            return self.next.image

    def progress_step(self, indexes, opencv_image):
        self.prev.image, self.prev.index = self.next.image, self.next.index
        self.next.image, self.next.index = opencv_image, indexes.frame.i
        return self.steps

    def _set_up_step_vars(self, data):
        # determine last frame and frame to start on
        prev_frame, next_frame, prev_img, next_img = call_get_resume_vars(data, self)
        if self.steps > 1:
            self.prev.image, self.prev.index = prev_img, prev_frame if prev_frame >= 0 else 0
            self.next.image, self.next.index = next_img, next_frame if next_frame >= 0 else 0

    def find_start(self, data) -> int:
        """Maybe resume animation (requires at least two frames - see function)."""
        # set start_frame to next frame
        if data.is_resuming_from_timestring():
            self._set_up_step_vars(data)
            return self.next.index + 1
        else:
            return 0

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

    def is_first_step_with_subtitles(self, render_data) -> bool:
        return self.is_first_step() and Srt.is_subtitle_generation_active(render_data.args.opts.data)

    def is_emit_in_between_frames(self) -> bool:
        return self.steps > 1
