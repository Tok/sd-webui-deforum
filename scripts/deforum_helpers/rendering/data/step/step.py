from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..initialization import RenderInit
from ..schedule import Schedule
from ..turbo import Turbo
from ..images import Images
from ..indexes import Indexes
from ..mask import Mask
from ...util import memory_utils, opt_utils, web_ui_utils
from ...util.call.anim import call_anim_frame_warp
from ...util.call.hybrid import (
    call_get_flow_for_hybrid_motion, call_get_flow_for_hybrid_motion_prev,
    call_get_matrix_for_hybrid_motion,
    call_get_matrix_for_hybrid_motion_prev, call_hybrid_composite)
from ...util.call.images import call_add_noise
from ...util.call.mask import call_compose_mask_with_check, call_unsharp_mask
from ...util.call.subtitle import call_format_animation_params, call_write_frame_subtitle
from ....hybrid_video import image_transform_ransac, image_transform_optical_flow
from ...util.log_utils import print_animation_frame_info


@dataclass(init=True, frozen=True, repr=False, eq=False)
class StepInit:
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
        return StepInit(keys.noise_schedule_series[i],
                        keys.strength_schedule_series[i],
                        keys.cfg_scale_schedule_series[i],
                        keys.contrast_schedule_series[i],
                        int(keys.kernel_schedule_series[i]),
                        keys.sigma_schedule_series[i],
                        keys.amount_schedule_series[i],
                        keys.threshold_schedule_series[i],
                        keys.cadence_flow_factor_schedule_series[i],
                        keys.redo_flow_factor_schedule_series[i],
                        StepInit._hybrid_comp_args(keys, i))

    @staticmethod
    def _hybrid_comp_args(keys, i):
        return {
            "alpha": keys.hybrid_comp_alpha_schedule_series[i],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[i],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[i],
            "mask_auto_contrast_cutoff_low": int(keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[i]),
            "mask_auto_contrast_cutoff_high": int(keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[i]),
            "flow_factor": keys.hybrid_flow_factor_schedule_series[i]}


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Step:
    init: StepInit
    schedule: Schedule
    depth: Any  # TODO try to init early, then freeze class
    subtitle_params_to_print: Any
    subtitle_params_string: str

    @staticmethod
    def do_start_and_create(init):
        # Perform the necessary side effects
        memory_utils.handle_med_or_low_vram_before_step(init)
        print_animation_frame_info(init)
        web_ui_utils.update_job(init)
        # Actual create
        step_init = StepInit.create(init.animation_keys.deform_keys, init.indexes.frame.i)
        schedule = Schedule.create(init, init.indexes.frame.i, init.args.anim_args, init.args.args)
        return Step(step_init, schedule, None, None, "")

    def is_optical_flow_redo_before_generation(self, optical_flow_redo_generation, images):
        has_flow_redo = optical_flow_redo_generation != 'None'
        return has_flow_redo and images.has_previous() and self.init.has_strength()

    def update_depth_prediction(self, init: RenderInit, turbo: Turbo):
        has_depth = init.depth_model is not None
        has_next = turbo.next.image is not None
        if has_depth and has_next:
            image = turbo.next.image
            weight = init.args.anim_args.midas_weight
            precision = init.args.root.half_precision
            self.depth = init.depth_model.predict(image, weight, precision)

    def write_frame_subtitle(self, init):
        if init.turbo.is_first_step_with_subtitles(init):
            self.subtitle_params_to_print = opt_utils.generation_info_for_subtitles(init)
            self.subtitle_params_string = call_format_animation_params(init, init.indexes.frame.i, params_to_print)
            call_write_frame_subtitle(init, init.indexes.frame.i, params_string)

    def write_frame_subtitle_if_active(self, init):
        if opt_utils.is_generate_subtitles(init):
            self.subtitle_params_to_print = opt_utils.generation_info_for_subtitles(init)
            self.subtitle_params_string = call_format_animation_params(init, self.indexes.tween.i, params_to_print)
            call_write_frame_subtitle(init, self.indexes.tween.i, params_string, sub_step.tween < 1.0)

    def apply_frame_warp_transform(self, init, image):
        previous, self.depth = call_anim_frame_warp(init, init.indexes.frame.i, image, None)
        return previous

    def _do_hybrid_compositing_on_cond(self, init, image, condition):
        i = init.indexes.frame.i
        schedules = self.init.hybrid_comp_schedules
        if condition:
            _, composed = call_hybrid_composite(init, i, image, schedules)
            return composed
        else:
            return image

    def do_hybrid_compositing_before_motion(self, init, image):
        condition = init.is_hybrid_composite_before_motion()
        return self._do_hybrid_compositing_on_cond(init, image, condition)

    def do_normal_hybrid_compositing_after_motion(self, init, image):
        condition = init.is_normal_hybrid_composite()
        return self._do_hybrid_compositing_on_cond(init, image, condition)

    def apply_scaling(self, image):
        return (image * self.init.contrast).round().astype(np.uint8)

    def apply_anti_blur(self, init, image):
        if self.init.amount > 0:
            return call_unsharp_mask(init, self, image, init.mask)
        else:
            return image

    def apply_frame_noising(self, init, mask, image):
        is_use_any_mask = init.args.args.use_mask or init.args.anim_args.use_noise_mask
        if is_use_any_mask:
            seq = self.schedule.noise_mask_seq
            vals = mask.noise_vals
            init.args.root.noise_mask = call_compose_mask_with_check(init, seq, vals, contrast_image)
        return call_add_noise(init, self, image)

    @staticmethod
    def apply_color_matching(init, image):
        if init.has_color_coherence():
            if init.images.color_match is None:
                # TODO questionable
                # initialize color_match for next iteration with current image, but don't do anything yet.
                init.images.color_match = image.copy()
            else:
                return maintain_colors(image, init.images.color_match, init.args.anim_args.color_coherence)
        return image

    @staticmethod
    def transform_to_grayscale_if_active(init, image):
        if init.args.anim_args.color_force_grayscale:
            grayscale = cv2.cvtColor(init.images.previous, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        else:
            return image

    @staticmethod
    def apply_hybrid_motion_ransac_transform(init, image):
        """hybrid video motion - warps images.previous to match motion, usually to prepare for compositing"""
        motion = init.args.anim_args.hybrid_motion
        if motion in ['Affine', 'Perspective']:
            last_i = init.indexes.frame.i - 1
            reference_images = init.images
            matrix = call_get_matrix_for_hybrid_motion_prev(init, last_i, reference_images.previous) \
                if init.args.anim_args.hybrid_motion_use_prev_img \
                else call_get_matrix_for_hybrid_motion(init, last_i)
            return image_transform_ransac(image, matrix, init.args.anim_args.hybrid_motion)
        return image

    @staticmethod
    def apply_hybrid_motion_optical_flow(init, image):
        motion = init.args.anim_args.hybrid_motion
        if motion in ['Optical Flow']:
            last_i = init.indexes.frame.i - 1
            reference_images = init.images
            flow = call_get_flow_for_hybrid_motion_prev(init, last_i, reference_images.previous) \
                if init.args.anim_args.hybrid_motion_use_prev_img \
                else call_get_flow_for_hybrid_motion(init, last_i)
            transformed = image_transform_optical_flow(images.previous, flow, step.init.flow_factor())
            init.animation_mode.prev_flow = flow  # side effect
            return transformed
        else:
            return image

    @staticmethod
    def create_color_match_for_video(init):
        if init.args.anim_args.color_coherence == 'Video Input' and init.is_hybrid_available():
            if int(init.indexes.frame.i) % int(init.args.anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(preview_video_image_path(init, init.indexes))
                prev_vid_img = prev_vid_img.resize(init.dimensions(), PIL.Image.LANCZOS)
                init.images.color_match = np.asarray(prev_vid_img)
                return cv2.cvtColor(init.images.color_match, cv2.COLOR_RGB2BGR)
        return None
