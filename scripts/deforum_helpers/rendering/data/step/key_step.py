import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, List

import cv2
import numpy as np

from .tween_step import Tween
from ..render_data import RenderData
from ..schedule import Schedule
from ...util import log_utils, memory_utils, opt_utils
from ...util.call.anim import call_anim_frame_warp
from ...util.call.gen import call_generate
from ...util.call.hybrid import (
    call_get_flow_for_hybrid_motion, call_get_flow_for_hybrid_motion_prev, call_get_matrix_for_hybrid_motion,
    call_get_matrix_for_hybrid_motion_prev, call_hybrid_composite)
from ...util.call.images import call_add_noise
from ...util.call.mask import call_compose_mask_with_check, call_unsharp_mask
from ...util.call.subtitle import call_format_animation_params, call_write_frame_subtitle
from ...util.call.video_and_audio import call_render_preview
from ....hybrid_video import image_transform_ransac, image_transform_optical_flow
from ....save_images import save_image
from ....seed import next_seed


@dataclass(init=True, frozen=True, repr=False, eq=False)
class KeyStepData:
    noise: Any = None
    strength: Any = None
    scale: Any = None
    contrast: Any = None
    kernel: int = 0
    sigma: Any = None
    amount: Any = None
    threshold: Any = None
    cadence_flow_factor: Any = None
    redo_flow_factor: Any = None
    hybrid_comp_schedules: Any = None

    def kernel_size(self) -> tuple[int, int]:
        return self.kernel, self.kernel

    def flow_factor(self):
        return self.hybrid_comp_schedules['flow_factor']

    def has_strength(self):
        return self.strength > 0

    @staticmethod
    def create(deform_keys, i):
        keys = deform_keys
        return KeyStepData(
            keys.noise_schedule_series[i],
            keys.strength_schedule_series[i],
            keys.cfg_scale_schedule_series[i],
            keys.contrast_schedule_series[i],
            int(keys.kernel_schedule_series[i]),
            keys.sigma_schedule_series[i],
            keys.amount_schedule_series[i],
            keys.threshold_schedule_series[i],
            keys.cadence_flow_factor_schedule_series[i],
            keys.redo_flow_factor_schedule_series[i],
            KeyStepData._hybrid_comp_args(keys, i))

    @staticmethod
    def _hybrid_comp_args(keys, i):
        return {
            "alpha": keys.hybrid_comp_alpha_schedule_series[i],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[i],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[i],
            "mask_auto_contrast_cutoff_low": int(keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[i]),
            "mask_auto_contrast_cutoff_high": int(keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[i]),
            "flow_factor": keys.hybrid_flow_factor_schedule_series[i]}


class KeyIndexDistribution(Enum):
    UNIFORM_SPACING = "Uniform Spacing"
    RANDOM_SPACING = "Random Spacing"
    RANDOM_PLACEMENT = "Random Placement"

    @property
    def name(self):
        return self.value

    @staticmethod
    def default():
        return KeyIndexDistribution.UNIFORM_SPACING


@dataclass(init=True, frozen=False, repr=False, eq=False)
class KeyStep:
    """Key steps are the steps for frames that actually get diffused (as opposed to tween frame steps)."""
    i: int
    step_data: KeyStepData
    render_data: RenderData
    schedule: Schedule
    depth: Any  # TODO try to init early, then freeze class
    subtitle_params_to_print: Any
    subtitle_params_string: str
    last_preview_frame: int
    tweens: List[Tween]
    tween_values: List[float]

    @staticmethod
    def create(data: RenderData):
        step_data = KeyStepData.create(data.animation_keys.deform_keys, data.indexes.frame.i)
        schedule = Schedule.create(data, data.indexes.frame.i, data.args.anim_args, data.args.args)
        return KeyStep(0, step_data, data, schedule, None, None, "", 0, list(), list())

    @staticmethod
    def create_all_steps(data, start_index, index_dist: KeyIndexDistribution = KeyIndexDistribution.default()):
        """Creates a list of key steps for the entire animation."""
        max_frames = data.args.anim_args.max_frames
        max_steps = 1 + int((max_frames - start_index) / data.cadence())
        log_utils.debug(f"max_steps {max_steps} max_frames {max_frames}")
        key_steps = [KeyStep.create(data) for _ in range(0, max_steps)]
        key_steps = KeyStep.recalculate_and_check_tweens(key_steps, start_index, max_frames, max_steps, index_dist)

        # Print message  # TODO move to log_utils
        tween_count = sum(len(key_step.tweens) for key_step in key_steps) - len(key_steps)
        msg_start = f"Created {len(key_steps)} KeySteps with {tween_count} Tween frames."
        msg_end = f"Key frame index distribution: '{index_dist.name}'."
        log_utils.info(f"{msg_start} {msg_end}")
        return key_steps

    @staticmethod
    def recalculate_and_check_tweens(key_steps, start_index, max_frames, max_steps, index_distribution):
        key_steps = KeyStep._recalculate_and_index_key_steps(key_steps, start_index, max_frames, index_distribution)
        key_steps = KeyStep._add_tweens_to_key_steps(key_steps)
        assert len(key_steps) == max_steps

        # FIXME seems to generate an unnecessary additional tween for value 1.0 (overwritten by actual frame?)
        #log_utils.debug("tween count: " + str(sum(len(key_step.tweens) for key_step in key_steps)))
        #expected_total_count = max_frames  # Total should be equal to max_frames
        #actual_total_count = len(key_steps) + sum(len(key_step.tweens) for key_step in key_steps)
        #log_utils.debug("total count: " + str(actual_total_count))
        #assert actual_total_count == expected_total_count  # Check total matches expected

        assert key_steps[0].i == 1
        assert key_steps[-1].i == max_frames
        return key_steps

    @staticmethod
    def _recalculate_and_index_key_steps(key_steps: List['KeyStep'], start_index, max_frames, index_distribution):
        def calculate_uniform_indexes():
            return [1 + start_index + int(n * (max_frames - 1 - start_index) / (len(key_steps) - 1))
                    for n in range(len(key_steps))]

        # TODO move logic into enum
        if index_distribution == KeyIndexDistribution.UNIFORM_SPACING:
            uniform_indexes = calculate_uniform_indexes()
            for i, key_step in enumerate(key_steps):
                key_step.i = uniform_indexes[i]
        elif index_distribution == KeyIndexDistribution.RANDOM_SPACING:
            uniform_indexes = calculate_uniform_indexes()
            key_steps[0].i = start_index + 1  # Enforce first index
            key_steps[-1].i = max_frames  # Enforce last index
            total_spacing = 0  # Calculate initial average spacing
            for i in range(1, len(key_steps)):
                total_spacing += key_steps[i].i - key_steps[i - 1].i
            average_spacing = total_spacing / (len(key_steps) - 1)  # Avoid division by zero
            # Noise factor to control randomness (adjust as needed)
            noise_factor = 0.5  # Higher value creates more variation
            log_utils.debug(f"average_spacing {average_spacing}")
            for i, key_step in enumerate(key_steps):
                if i == 0 or i == len(key_steps) - 1:
                    continue  # Skip first and last (already set)
                base_index = uniform_indexes[i]
                noise = random.uniform(-noise_factor, noise_factor) * average_spacing
                log_utils.debug(f"base_index {base_index} noise {noise} i {int(base_index + noise)}")
                key_step.i = int(base_index + noise)  # Apply base index and noise
                # Ensure index stays within frame bounds
                key_step.i = max(start_index, min(key_step.i, max_frames - 1))
        elif index_distribution == KeyIndexDistribution.RANDOM_PLACEMENT:
            key_steps[0].i = start_index + 1  # Enforce first index
            key_steps[-1].i = max_frames  # Enforce last index
            # Randomly distribute indices for remaining keyframes
            for i in range(1, len(key_steps) - 1):
                key_step = key_steps[i]
                key_step.i = random.randint(start_index + 1, max_frames - 2)
        else:
            raise KeyError(f"KeyIndexDistribution {index_distribution} doesn't exist.")

        is_random = index_distribution in [KeyIndexDistribution.RANDOM_SPACING, KeyIndexDistribution.RANDOM_PLACEMENT]
        return key_steps.sort(key=lambda ks: ks.i) if is_random else key_steps

    @staticmethod
    def _add_tweens_to_key_steps(key_steps):
        for i in range(1, len(key_steps)):  # skipping 1st key frame
            data = key_steps[i].render_data
            if data.turbo.is_emit_in_between_frames():
                from_i = key_steps[i-1].i
                to_i = key_steps[i].i
                tweens, values = Tween.create_in_between_steps(key_steps[i], data, from_i, to_i)
                for tween in tweens:  # TODO move to creation
                    tween.indexes.update_tween_index(tween.i() + key_steps[i].i)
                log_utils.info(f"Creating {len(tweens)} tween steps ({from_i}->{to_i}) for key step {key_steps[i].i}")
                key_steps[i].tweens = tweens
                key_steps[i].tween_values = values
                key_steps[i].render_data.indexes.update_tween_start(data.turbo)
        return key_steps

    def is_optical_flow_redo_before_generation(self, optical_flow_redo_generation, images):
        has_flow_redo = optical_flow_redo_generation != 'None'
        return has_flow_redo and images.has_previous() and self.step_data.has_strength()

    def maybe_write_frame_subtitle(self):
        data = self.render_data
        if data.turbo.is_first_step_with_subtitles(data):
            self.subtitle_params_to_print = opt_utils.generation_info_for_subtitles(data)
            self.subtitle_params_string = call_format_animation_params(data, data.indexes.frame.i, params_to_print)
            call_write_frame_subtitle(data, data.indexes.frame.i, params_string)

    def write_frame_subtitle_if_active(self, data: RenderData):
        if opt_utils.is_generate_subtitles(data):
            self.subtitle_params_to_print = opt_utils.generation_info_for_subtitles(data)
            self.subtitle_params_string = call_format_animation_params(data, self.indexes.tween.i, params_to_print)
            call_write_frame_subtitle(data, self.indexes.tween.i, params_string, sub_step.tween < 1.0)

    def apply_frame_warp_transform(self, data: RenderData, image):
        previous, self.depth = call_anim_frame_warp(data, self.i, image, None)
        return previous

    def _do_hybrid_compositing_on_cond(self, data: RenderData, image, condition):
        i = data.indexes.frame.i
        schedules = self.step_data.hybrid_comp_schedules
        if condition:
            _, composed = call_hybrid_composite(data, i, image, schedules)
            return composed
        else:
            return image

    def do_hybrid_compositing_before_motion(self, data: RenderData, image):
        condition = data.is_hybrid_composite_before_motion()
        return self._do_hybrid_compositing_on_cond(data, image, condition)

    def do_normal_hybrid_compositing_after_motion(self, data: RenderData, image):
        condition = data.is_normal_hybrid_composite()
        return self._do_hybrid_compositing_on_cond(data, image, condition)

    def apply_scaling(self, image):
        return (image * self.step_data.contrast).round().astype(np.uint8)

    def apply_anti_blur(self, data: RenderData, image):
        if self.step_data.amount > 0:
            return call_unsharp_mask(data, self, image, data.mask)
        else:
            return image

    def apply_frame_noising(self, data: RenderData, mask, image):
        is_use_any_mask = data.args.args.use_mask or data.args.anim_args.use_noise_mask
        if is_use_any_mask:
            seq = self.schedule.noise_mask_seq
            vals = mask.noise_vals
            data.args.root.noise_mask = call_compose_mask_with_check(data, seq, vals, contrast_image)
        return call_add_noise(data, self, image)

    @staticmethod
    def apply_color_matching(data: RenderData, image):
        if data.has_color_coherence():
            if data.images.color_match is None:
                # TODO questionable
                # initialize color_match for next iteration with current image, but don't do anything yet.
                data.images.color_match = image.copy()
            else:
                return maintain_colors(image, data.images.color_match, data.args.anim_args.color_coherence)
        return image

    @staticmethod
    def transform_to_grayscale_if_active(data: RenderData, image):
        if data.args.anim_args.color_force_grayscale:
            grayscale = cv2.cvtColor(data.images.previous, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        else:
            return image

    @staticmethod
    def apply_hybrid_motion_ransac_transform(data: RenderData, image):
        """hybrid video motion - warps `images.previous` to match motion, usually to prepare for compositing"""
        motion = data.args.anim_args.hybrid_motion
        if motion in ['Affine', 'Perspective']:
            last_i = data.indexes.frame.i - 1
            reference_images = data.images
            matrix = call_get_matrix_for_hybrid_motion_prev(data, last_i, reference_images.previous) \
                if data.args.anim_args.hybrid_motion_use_prev_img \
                else call_get_matrix_for_hybrid_motion(data, last_i)
            return image_transform_ransac(image, matrix, data.args.anim_args.hybrid_motion)
        return image

    @staticmethod
    def apply_hybrid_motion_optical_flow(data: RenderData, image):
        motion = data.args.anim_args.hybrid_motion
        if motion in ['Optical Flow']:
            last_i = data.indexes.frame.i - 1
            reference_images = data.images
            flow = call_get_flow_for_hybrid_motion_prev(data, last_i, reference_images.previous) \
                if data.args.anim_args.hybrid_motion_use_prev_img \
                else call_get_flow_for_hybrid_motion(data, last_i)
            transformed = image_transform_optical_flow(images.previous, flow, step.step_data.flow_factor())
            data.animation_mode.prev_flow = flow  # side effect
            return transformed
        else:
            return image

    def create_color_match_for_video(self):
        data = self.render_data
        if data.args.anim_args.color_coherence == 'Video Input' and data.is_hybrid_available():
            if int(data.indexes.frame.i) % int(data.args.anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(preview_video_image_path(data, data.indexes))
                prev_vid_img = prev_vid_img.resize(data.dimensions(), PIL.Image.LANCZOS)
                data.images.color_match = np.asarray(prev_vid_img)
                return cv2.cvtColor(data.images.color_match, cv2.COLOR_RGB2BGR)
        return None

    def transform_and_update_noised_sample(self, frame_tube, contrasted_noise_tube):
        data = self.render_data
        if data.images.has_previous():  # skipping 1st iteration
            transformed_image = frame_tube(data, self)(data.images.previous)
            # TODO separate
            noised_image = contrasted_noise_tube(data, self)(transformed_image)
            data.update_sample_and_args_for_current_progression_step(self, noised_image)
            return transformed_image
        else:
            return None

    def prepare_generation(self, frame_tube, contrasted_noise_tube):
        self.render_data.images.color_match = self.create_color_match_for_video()
        self.render_data.images.previous = self.transform_and_update_noised_sample(frame_tube, contrasted_noise_tube)
        self.render_data.prepare_generation(self.render_data, self)
        self.maybe_redo_optical_flow()
        self.maybe_redo_diffusion()

    # Conditional Redoes
    def maybe_redo_optical_flow(self):
        data = self.render_data
        optical_flow_redo_generation = data.optical_flow_redo_generation_if_not_in_preview_mode()
        is_redo_optical_flow = self.is_optical_flow_redo_before_generation(optical_flow_redo_generation, data.images)
        if is_redo_optical_flow:
            data.args.root.init_sample = self.do_optical_flow_redo_before_generation()

    def maybe_redo_diffusion(self):
        data = self.render_data
        is_pos_redo = data.has_positive_diffusion_redo
        is_diffusion_redo = is_pos_redo and data.images.has_previous() and self.step_data.has_strength()
        is_not_preview = data.is_not_in_motion_preview_mode()
        if is_diffusion_redo and is_not_preview:
            self.do_diffusion_redo()

    def do_generation(self):
        return call_generate(self.render_data, self)

    def progress_and_save(self, image):
        next_index = self._progress_save_and_get_next_index(image)
        self.render_data.indexes.update_frame(next_index)

    def _progress_save_and_get_next_index(self, image):
        data = self.render_data
        """Will progress frame or turbo-frame step, save the image, update `self.depth` and return next index."""
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not data.animation_mode.has_video_input:
            data.images.previous = opencv_image
        if data.turbo.has_steps():
            return data.indexes.frame.i + data.turbo.progress_step(data.indexes, opencv_image)
        else:
            filename = filename_utils.frame(data, data.indexes)
            save_image(image, 'PIL', filename, data.args.args, data.args.video_args, data.args.root)
            self.depth = generate_depth_maps_if_active(data)
            return data.indexes.frame.i + 1  # normal (i.e. 'non-turbo') step always increments by 1.

    def next_seed(self):
        return next_seed(self.render_data.args.args, self.render_data.args.root)

    def update_render_preview(self):
        self.last_preview_frame = call_render_preview(self.render_data, self.last_preview_frame)

    def generate_depth_maps_if_active(self):
        data = self.render_data
        # TODO move all depth related stuff to new class.
        if data.args.anim_args.save_depth_maps:
            memory_utils.handle_vram_before_depth_map_generation(data)
            depth = data.depth_model.predict(opencv_image, data.args.anim_args.midas_weight,
                                             data.args.root.half_precision)
            depth_filename = filename_utils.depth_frame(data, idx)
            data.depth_model.save(os.path.join(data.output_directory, depth_filename), depth)
            memory_utils.handle_vram_after_depth_map_generation(data)
            return depth

    def do_optical_flow_redo_before_generation(self):
        data = self.render_data
        stored_seed = data.args.args.seed  # keep original to reset it after executing the optical flow
        data.args.args.seed = generate_random_seed()  # set a new random seed
        print_optical_flow_info(data, optical_flow_redo_generation)  # TODO output temp seed?

        sample_image = call_generate(data, data.indexes.frame.i)
        optical_tube = optical_flow_redo_tube(data, optical_flow_redo_generation)
        transformed_sample_image = optical_tube(sample_image)

        data.args.args.seed = stored_seed  # restore stored seed
        return Image.fromarray(transformed_sample_image)

    def do_diffusion_redo(self):
        data = self.render_data
        stored_seed = data.args.args.seed
        last_diffusion_redo_index = int(data.args.anim_args.diffusion_redo)
        for n in range(0, last_diffusion_redo_index):
            print_redo_generation_info(data, n)
            data.args.args.seed = generate_random_seed()
            diffusion_redo_image = call_generate(data, self)
            diffusion_redo_image = cv2.cvtColor(np.array(diffusion_redo_image), cv2.COLOR_RGB2BGR)
            # color match on last one only
            is_last_iteration = n == last_diffusion_redo_index
            if is_last_iteration:
                mode = data.args.anim_args.color_coherence
                diffusion_redo_image = maintain_colors(data.images.previous, data.images.color_match, mode)
            data.args.args.seed = stored_seed
            data.args.root.init_sample = Image.fromarray(cv2.cvtColor(diffusion_redo_image, cv2.COLOR_BGR2RGB))
