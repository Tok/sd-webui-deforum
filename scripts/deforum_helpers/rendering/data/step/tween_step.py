from dataclasses import dataclass
from typing import Any

from ...util import image_utils, log_utils, web_ui_utils


@dataclass(init=True, frozen=False, repr=False, eq=False)
class TweenStep:
    """cadence vars"""
    tween: float
    cadence_flow: Any
    cadence_flow_inc: Any

    @staticmethod
    def _calculate_tween_from_indices(frame_difference, last_step) -> float:
        return min(0.0, max(1.0, float(last_step) / float(frame_difference)))

    @staticmethod
    def create(indexes):
        tween = float(indexes.tween.i - indexes.tween.start + 1) / float(indexes.frame.i - indexes.tween.start)
        return TweenStep(tween, None, None)

    @staticmethod
    def create_directly(from_index, to_index):
        tween = TweenStep._calculate_tween_from_indices(from_index, to_index)
        return TweenStep(tween, None, None)

    def generate_tween_image(self, init, indexes, step, turbo):
        is_tween = True
        warped = turbo.do_optical_flow_cadence_after_animation_warping(init, indexes, step, self)
        recolored = image_utils.force_tween_to_grayscale_if_required(init, warped)
        masked = image_utils.add_overlay_mask_if_active(init, recolored, is_tween)
        return masked

    @staticmethod
    def process(init, indexes, step, turbo, images, tween_step):
        turbo.advance_optical_flow_cadence_before_animation_warping(init, tween_step)
        step.update_depth_prediction(init, turbo)
        turbo.advance(init, indexes.tween.i, step.depth)
        turbo.do_hybrid_video_motion(init, indexes, images)

    @staticmethod
    def generate_and_save_frame(init, indexes, step, turbo, tween_step, is_do_save: bool = True):
        new_image = tween_step.generate_tween_image(init, indexes, step, turbo)
        if is_do_save:
            image_utils.save_cadence_frame_and_depth_map_if_active(init, indexes, new_image)
        return new_image

    @staticmethod
    def handle_synchronous_status_concerns(init, indexes, step, tween_step):
        step.write_frame_subtitle_if_active(init, indexes)  # TODO decouple from execution and calc all in advance?
        log_utils.print_tween_frame_info(init, indexes, tween_step.cadence_flow, tween_step.tween)
        web_ui_utils.update_progress_during_cadence(init, indexes)
