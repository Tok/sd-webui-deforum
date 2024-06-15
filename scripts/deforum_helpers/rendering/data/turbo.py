from dataclasses import dataclass

from cv2.typing import MatLike

from .subtitle import Srt
from ..util.call.anim import call_anim_frame_warp
from ..util.call.resume import call_get_resume_vars
from ...hybrid_video import image_transform_ransac, image_transform_optical_flow


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

    def do_hybrid_video_motion(self, init, indexes, reference_images):
        """Warps the previous and/or the next to match the motion of the provided reference images."""
        motion = init.args.anim_args.hybrid_motion

        def _is_do_motion(motions):
            return indexes.tween.i > 0 and motion in motions

        if _is_do_motion(['Affine', 'Perspective']):
            self.advance_hybrid_motion_ransac_trasform(init, indexes, reference_images)
        if _is_do_motion(['Optical Flow']):
            self.advance_hybrid_motion_optical_tween_flow(init, indexes, reference_images, step)

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

    def advance_hybrid_motion_optical_tween_flow(self, init, indexes, reference_images, step):
        if init.args.anim_args.hybrid_motion_use_prev_img:
            flow = call_get_flow_for_hybrid_motion_prev(init, indexes.tween.i - 1, reference_images.previous)
            turbo.advance_optical_tween_flow(self, step, flow)
            init.animation_mode.prev_flow = flow
        else:
            flow = call_get_flow_for_hybrid_motion(init, indexes.tween.i - 1)
            turbo.advance_optical_tween_flow(self, step, flow)
            init.animation_mode.prev_flow = flow

    def advance_cadence_flow(self, tween_step):
        ff = step.step_data.sub_step.cadence_flow_factor
        i = indexes.tween.i
        inc = tween_step.cadence_flow_inc
        if self.is_advance_prev(i):
            self.prev.image = image_transform_optical_flow(self.prev.image, inc, ff)
        if self.is_advance_next(i):
            self.next.image = image_transform_optical_flow(self.next.image, inc, ff)

    def advance_ransac_trasform(self, init, matrix):
        i = indexes.tween.i
        motion = init.args.anim_args.hybrid_motion
        if self.is_advance_prev(i):
            self.prev.image = image_transform_ransac(self.prev.image, matrix, motion)
        if self.is_advance_next(i):
            self.next.image = image_transform_ransac(self.next.image, matrix, motion)

    def advance_hybrid_motion_ransac_trasform(self, init, indexes, reference_images):
        if init.args.anim_args.hybrid_motion_use_prev_img:
            matrix = call_get_matrix_for_hybrid_motion_prev(init, indexes.tween.i - 1, reference_images.previous)
            turbo.advance_ransac_trasform(init, matrix)
        else:
            matrix = call_get_matrix_for_hybrid_motion(init, indexes.tween.i - 1)
            turbo.advance_ransac_trasform(init, matrix)

    def advance_optical_flow_cadence_before_animation_warping(self, init, tween_step):
        if init.is_3d_or_2d() and init.has_optical_flow_cadence():
            has_tween_schedule = init.animation_keys.deform_keys.strength_schedule_series[indexes.tween.start.i] > 0
            has_images = self.prev.image is not None and self.next.image is not None
            has_step_and_images = tween_step.cadence_flow is None and has_images
            if has_tween_schedule and has_step_and_images:
                cadence = init.args.anim_args.optical_flow_cadence
                flow = call_get_flow_from_images(init, self.prev.image, self.next.image, cadence)
                tween_step.cadence_flow = (flow / 2)
            self.next.image = advance_optical_flow
            self.advance_optical_flow(tween_step)
            self.next.image = image_transform_optical_flow(self.next.image, -tween_step.cadence_flow)

    def do_optical_flow_cadence_after_animation_warping(self, init, indexes, step, tween_step):
        if tween_step.cadence_flow is not None:
            # TODO Calculate all increments before running the generation (and try to avoid abs->rel->abc conversions).
            temp_flow = abs_flow_to_rel_flow(tween_step.cadence_flow, init.width(), init.height())
            new_flow, _ = call_anim_frame_warp(init, indexes.tween.i, temp_flow, step.depth)
            tween_step.cadence_flow = new_flow
            abs_flow = rel_flow_to_abs_flow(tween_step.cadence_flow, init.width(), init.height())
            tween_step.cadence_flow_inc = abs_flow * tween_step.tween
            self.advance_cadence_flow(tween_step)
        self.prev.index = self.next.frame_idx = indexes.tween.i
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
