from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable, Tuple

from ..turbo import Turbo
from ...data.indexes import Indexes, IndexWithStart
from ...data.render_data import RenderData
from ...util import image_utils, log_utils, opt_utils, web_ui_utils
from ...util.call.subtitle import call_format_animation_params, call_write_frame_subtitle


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Tween:
    """cadence vars"""
    indexes: Indexes
    tween: float
    cadence_flow: Any  # late init
    cadence_flow_inc: Any  # late init
    depth: Any
    depth_prediction: Any

    def i(self):
        return self.indexes.tween.i

    def from_key_step_i(self):
        return self.indexes.frame.start

    def to_key_step_i(self):
        return self.indexes.frame.i

    def emit_frame(self, last_step, grayscale_tube, overlay_mask_tube):
        """Emits this tween frame."""
        max_frames = last_step.render_data.args.anim_args.max_frames
        if self.i() >= max_frames - 1:
            return

        # log_utils.debug(f"Emitting tween frame {self.i()} for step {last_step.i}")
        data = last_step.render_data
        self.handle_synchronous_status_concerns(data)
        self.process(data)

        new_image = self.generate(data, grayscale_tube, overlay_mask_tube)
        # TODO pass step and depth instead of data and tween_step.indexes
        new_image = image_utils.save_and_return_frame(data, self.i(), new_image)

        # updating reference images to calculate hybrid motions in next iteration
        data.images.previous = new_image  # FIXME

    @staticmethod
    def create_in_between_steps(last_step, data, from_i, to_i):
        tween_range = range(from_i, to_i)
        tween_indexes_list: List[Indexes] = Tween.create_indexes(data.indexes, tween_range)
        tween_steps_and_values = Tween.create_steps(last_step, tween_indexes_list)
        return tween_steps_and_values

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
        indexes_list = [Tween._increment(last_step.render_data.indexes.copy(), count, i + 1) for i in r]
        return list((Tween(indexes_list[i], values[i], None, None, last_step.depth, None) for i in r))

    @staticmethod
    def create_indexes(base_indexes: Indexes, frame_range: Iterable[int]) -> list[Indexes]:
        return list(chain.from_iterable([Indexes.create_from_last(base_indexes, i)] for i in frame_range))

    @staticmethod
    def create_steps(last_step, tween_indexes_list: list[Indexes]) -> Tuple[list['Tween'], list[float]]:
        if len(tween_indexes_list) > 0:
            expected_tween_frames = Tween._calculate_expected_tween_frames(len(tween_indexes_list))
            return Tween.create_steps_from_values(last_step, expected_tween_frames), expected_tween_frames
        else:
            return list(), list()

    def generate_tween_image(self, data, grayscale_tube, overlay_mask_tube):
        is_tween = True
        warped = data.turbo.do_optical_flow_cadence_after_animation_warping(data, self.indexes, self)
        recolored = grayscale_tube(data)(warped)
        masked = overlay_mask_tube(data, is_tween)(recolored)
        return masked

    def generate(self, data, grayscale_tube, overlay_mask_tube):
        return self.generate_tween_image(data, grayscale_tube, overlay_mask_tube)

    @staticmethod
    def calculate_depth_prediction(data, turbo: Turbo):
        has_depth = data.depth_model is not None
        has_next = turbo.next.image is not None
        if has_depth and has_next:
            image = turbo.next.image
            weight = data.args.anim_args.midas_weight
            precision = data.args.root.half_precision
            return data.depth_model.predict(image, weight, precision)

    def process(self, data):
        data.turbo.advance_optical_flow_cadence_before_animation_warping(data, self)
        self.depth_prediction = Tween.calculate_depth_prediction(data, data.turbo)
        data.turbo.advance(data, self.indexes.tween.i, self.depth)
        data.turbo.do_hybrid_video_motion(data, self.indexes, data.images)  # TODO remove self.indexes or init.indexes

    def handle_synchronous_status_concerns(self, data):
        self.write_tween_frame_subtitle_if_active(data)  # TODO decouple from execution and calc all in advance?
        log_utils.print_tween_frame_info(data, self.indexes, self.cadence_flow, self.tween)
        web_ui_utils.update_progress_during_cadence(data, self.indexes)

    def write_tween_frame_subtitle_if_active(self, data: RenderData):
        if opt_utils.is_generate_subtitles(data):
            params_to_print = opt_utils.generation_info_for_subtitles(data)
            params_string = call_format_animation_params(data, self.indexes.tween.i, params_to_print)
            call_write_frame_subtitle(data, self.indexes.tween.i, params_string, sub_step.tween < 1.0)

    @staticmethod
    def maybe_emit_in_between_frames(last_step, grayscale_tube, overlay_mask_tube):
        # TODO? return the new frames
        if last_step.render_data.turbo.is_emit_in_between_frames():
            diff = last_step.render_data.indexes.frame.i - last_step.render_data.turbo.steps
            tween_frame_start_i = max(last_step.render_data.indexes.frame.start, diff)
            from_i = tween_frame_start_i
            to_i = last_step.render_data.indexes.frame.i
            return Tween.emit_frames_between_index_pair(last_step, from_i, to_i, grayscale_tube, overlay_mask_tube)
        return last_step

    @staticmethod
    def emit_frames_between_index_pair(last_step, from_i, to_i, grayscale_tube, overlay_mask_tube):
        """Emits tween frames (also known as turbo- or cadence-frames) between the provided indices."""
        tween_range = range(from_i, to_i)
        tween_indexes_list: List[Indexes] = Tween.create_indexes(last_step.render_data.indexes, tween_range)
        tween_steps, values = Tween.create_steps(last_step, tween_indexes_list)
        last_step.render_data.indexes.update_tween_start(last_step.render_data.turbo)
        log_utils.print_tween_frame_from_to_info(last_step.render_data.turbo.steps, values, from_i, to_i)
        return Tween.emit_tween_frames(last_step, tween_steps, grayscale_tube, overlay_mask_tube)

    @staticmethod
    def emit_tween_frames(last_step, tween_steps, grayscale_tube, overlay_mask_tube):
        """Emits a tween frame for each provided tween_step."""
        for tween_step in tween_steps:
            tween_step.handle_synchronous_status_concerns(last_step.render_data)
            tween_step.process(last_step.render_data)  # side effects on turbo and on step
            new_image = tween_step.generate(last_step.render_data, last_step.depth, grayscale_tube, overlay_mask_tube)
            # TODO pass step and depth instead of data and tween_step.indexes
            i = tween_step.indexes.tween.i
            new_image = image_utils.save_and_return_frame(last_step.render_data, i, new_image)
            # updating reference images to calculate hybrid motions in next iteration
            last_step.render_data.images.previous = new_image
        return last_step
