from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable

from ...data.indexes import Indexes
from ...data.step import Step
from ...img_2_img_tubes import conditional_force_tween_to_grayscale_tube, conditional_add_overlay_mask_tube
from ...util import image_utils, log_utils, web_ui_utils


@dataclass(init=True, frozen=False, repr=False, eq=False)
class TweenStep:
    """cadence vars"""
    indexes: Indexes
    last_step: Step
    tween: float
    cadence_flow: Any
    cadence_flow_inc: Any

    @staticmethod
    def _calculate_tween_from_indices(frame_difference, last_step) -> float:
        return min(0.0, max(1.0, float(last_step) / float(frame_difference)))

    @staticmethod
    def create(indexes, last_step):
        tween = float(indexes.tween.i - indexes.tween.start + 1) / float(indexes.frame.i - indexes.tween.start)
        return TweenStep(indexes, last_step, tween, None, None)

    @staticmethod
    def create_indexes(base_indexes: Indexes, frame_range: Iterable[int]) -> list[Indexes]:
        return list(chain.from_iterable([Indexes.create_from_last(base_indexes, i)] for i in frame_range))

    @staticmethod
    def create_steps(last_step, tween_indexes_list: list[Indexes]) -> list['TweenStep']:
        return [TweenStep.create(i.create_next(), last_step) for i in tween_indexes_list]

    @staticmethod
    def create_directly(from_index, to_index):
        tween = TweenStep._calculate_tween_from_indices(from_index, to_index)
        return TweenStep(tween, None, None)

    def generate_tween_image(self, init):
        is_tween = True
        warped = init.turbo.do_optical_flow_cadence_after_animation_warping(init, self.indexes, self.last_step, self)
        recolored = conditional_force_tween_to_grayscale_tube(init)(warped)
        masked = conditional_add_overlay_mask_tube(init, is_tween)(recolored)
        return masked

    def generate(self, init):
        return self.generate_tween_image(init)

    def process(self, init):
        init.turbo.advance_optical_flow_cadence_before_animation_warping(init, self)
        self.last_step.update_depth_prediction(init, init.turbo)
        init.turbo.advance(init, self.indexes.tween.i, self.last_step.depth)
        init.turbo.do_hybrid_video_motion(init, self.indexes, init.images)  # TODO remove self.indexes or init.indexes

    def handle_synchronous_status_concerns(self, init):
        self.last_step.write_frame_subtitle_if_active(init)  # TODO decouple from execution and calc all in advance?
        log_utils.print_tween_frame_info(init, self.indexes, self.cadence_flow, self.tween)
        web_ui_utils.update_progress_during_cadence(init, self.indexes)

    @staticmethod
    def maybe_emit_in_between_frames(step: Step):
        # TODO? return the new frames
        data = step.render_data
        if data.turbo.is_emit_in_between_frames():
            tween_frame_start_i = max(data.indexes.frame.start, data.indexes.frame.i - data.turbo.steps)
            TweenStep.emit_frames_between_index_pair(step, tween_frame_start_i, data.indexes.frame.i)

    @staticmethod
    def emit_frames_between_index_pair(step: Step, tween_frame_start_i, frame_i):
        """Emits tween frames (also known as turbo- or cadence-frames) between the provided indices."""
        tween_range = range(tween_frame_start_i, frame_i)
        tween_indexes_list: List[Indexes] = TweenStep.create_indexes(step.render_data.indexes, tween_range)
        tween_steps: List[TweenStep] = TweenStep.create_steps(step, tween_indexes_list)
        step.render_data.indexes.update_tween_start(step.render_data.turbo)
        TweenStep.emit_tween_frames(step, tween_steps)

    @staticmethod
    def emit_tween_frames(step: Step, tween_steps):
        """Emits a tween frame for each provided tween_step."""
        for tween_step in tween_steps:
            tween_step.handle_synchronous_status_concerns(step.render_data)
            tween_step.process(step.render_data)  # side effects on turbo and on step

            new_image = tween_step.generate(step.render_data)
            # TODO pass step and depth instead of data and tween_step.indexes
            new_image = image_utils.save_and_return_frame(step.render_data, tween_step.indexes, new_image)
            # updating reference images to calculate hybrid motions in next iteration
            step.render_data.images.previous = new_image
