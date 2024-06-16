from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable

from ...data.indexes import Indexes, IndexWithStart
from ...data.step import Step
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
    def _calculate_expected_tween_frames(num_entries):
        if num_entries <= 0:
            raise ValueError("Number of entries must be positive")
        offset = 1.0 / num_entries
        positions = [offset + (i / num_entries) for i in range(num_entries)]
        return positions

    @staticmethod
    def _increment(original_indexes, tween_count, from_start):
        inc = original_indexes.frame.i - tween_count - original_indexes.tween.start + from_start
        original_indexes.tween = IndexWithStart(original_indexes.tween.start, original_indexes.tween.start + inc)
        return original_indexes

    @staticmethod
    def create_steps_from_values(last_step, values):
        count = len(values)
        r = range(count)
        indexes_list = [TweenStep._increment(last_step.render_data.indexes.copy(), count, i + 1) for i in r]
        return list((TweenStep(indexes_list[i], last_step, values[i], None, None) for i in r))

    @staticmethod
    def create_indexes(base_indexes: Indexes, frame_range: Iterable[int]) -> list[Indexes]:
        return list(chain.from_iterable([Indexes.create_from_last(base_indexes, i)] for i in frame_range))

    @staticmethod
    def create_steps(last_step, tween_indexes_list: list[Indexes]) -> list['TweenStep']:
        if len(tween_indexes_list) > 0:
            expected_tween_frames = TweenStep._calculate_expected_tween_frames(len(tween_indexes_list))
            return TweenStep.create_steps_from_values(last_step, expected_tween_frames)
        else:
            return list()

    def generate_tween_image(self, data, grayscale_tube, overlay_mask_tube):
        is_tween = True
        warped = data.turbo.do_optical_flow_cadence_after_animation_warping(data, self.indexes, self.last_step, self)
        recolored = grayscale_tube(data)(warped)
        masked = overlay_mask_tube(data, is_tween)(recolored)
        return masked

    def generate(self, data, grayscale_tube, overlay_mask_tube):
        return self.generate_tween_image(data, grayscale_tube, overlay_mask_tube)

    def process(self, init):
        init.turbo.advance_optical_flow_cadence_before_animation_warping(init, self)
        self.last_step.update_depth_prediction(init, init.turbo)
        init.turbo.advance(init, self.indexes.tween.i, self.last_step.depth)
        init.turbo.do_hybrid_video_motion(init, self.indexes, init.images)  # TODO remove self.indexes or init.indexes

    def handle_synchronous_status_concerns(self, data):
        self.last_step.write_frame_subtitle_if_active(data)  # TODO decouple from execution and calc all in advance?
        log_utils.print_tween_frame_info(data, self.indexes, self.cadence_flow, self.tween)
        web_ui_utils.update_progress_during_cadence(data, self.indexes)

    @staticmethod
    def maybe_emit_in_between_frames(step: Step, grayscale_tube, overlay_mask_tube):
        # TODO? return the new frames
        if step.render_data.turbo.is_emit_in_between_frames():
            tween_frame_start_i = max(step.render_data.indexes.frame.start,
                                      step.render_data.indexes.frame.i - step.render_data.turbo.steps)
            return TweenStep.emit_frames_between_index_pair(step, tween_frame_start_i,
                                                            step.render_data.indexes.frame.i,
                                                            grayscale_tube, overlay_mask_tube)
        return step

    @staticmethod
    def emit_frames_between_index_pair(step: Step, tween_frame_start_i, frame_i, grayscale_tube, overlay_mask_tube):
        """Emits tween frames (also known as turbo- or cadence-frames) between the provided indices."""
        tween_range = range(tween_frame_start_i, frame_i)
        tween_indexes_list: List[Indexes] = TweenStep.create_indexes(step.render_data.indexes, tween_range)
        tween_steps: List[TweenStep] = TweenStep.create_steps(step, tween_indexes_list)
        step.render_data.indexes.update_tween_start(step.render_data.turbo)
        log_utils.print_tween_frame_from_to_info(step.render_data.turbo.steps, tween_frame_start_i, frame_i)
        return TweenStep.emit_tween_frames(step, tween_steps, grayscale_tube, overlay_mask_tube)

    @staticmethod
    def emit_tween_frames(step: Step, tween_steps, grayscale_tube, overlay_mask_tube):
        """Emits a tween frame for each provided tween_step."""
        for tween_step in tween_steps:
            tween_step.handle_synchronous_status_concerns(step.render_data)
            tween_step.process(step.render_data)  # side effects on turbo and on step

            new_image = tween_step.generate(step.render_data, grayscale_tube, overlay_mask_tube)
            # TODO pass step and depth instead of data and tween_step.indexes
            new_image = image_utils.save_and_return_frame(step.render_data, tween_step.indexes, new_image)
            # updating reference images to calculate hybrid motions in next iteration
            step.render_data.images.previous = new_image
        return step
