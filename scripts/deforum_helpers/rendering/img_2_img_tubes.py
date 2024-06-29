from typing import Callable

import cv2
import numpy as np
from cv2.typing import MatLike

from .data.render_data import RenderData
from .data.step.key_step import KeyStep
from .util.call.hybrid import call_get_flow_from_images, call_hybrid_composite
from .util.fun_utils import tube
from ..colors import maintain_colors
from ..masks import do_overlay_mask

"""
This module provides functions for conditionally processing images through various transformations.
The `tube` function allows chaining these transformations together to create flexible image processing pipelines.
Easily experiment by changing, or changing the order of function calls in the tube without having to worry
about the larger context and without having to invent unnecessary names for intermediary processing results.

All functions within the tube take and return an image (`img` argument). They may (and must) pass through
the original image unchanged if a specific transformation is disabled or not required.

Example:
transformed_image = my_tube(arguments)(original_image)
"""

# ImageTubes are functions that take a MatLike image and return a newly processed (or the same unchanged) MatLike image.
ImageTube = Callable[[MatLike], MatLike]


def frame_transformation_tube(data: RenderData, step: KeyStep) -> ImageTube:
    # make sure `img` stays the last argument in each call.
    return tube(lambda img: step.apply_frame_warp_transform(data, img),
                lambda img: step.do_hybrid_compositing_before_motion(data, img),
                lambda img: KeyStep.apply_hybrid_motion_ransac_transform(data, img),
                lambda img: KeyStep.apply_hybrid_motion_optical_flow(data, img),
                lambda img: step.do_normal_hybrid_compositing_after_motion(data, img),
                lambda img: KeyStep.apply_color_matching(data, img),
                lambda img: KeyStep.transform_to_grayscale_if_active(data, img))


def contrast_transformation_tube(data: RenderData, step: KeyStep) -> ImageTube:
    return tube(lambda img: step.apply_scaling(img),
                lambda img: step.apply_anti_blur(data, img))


def noise_transformation_tube(data: RenderData, step: KeyStep) -> ImageTube:
    return tube(lambda img: step.apply_frame_noising(data, step, img))


def optical_flow_redo_tube(data: RenderData, optical_flow) -> ImageTube:
    return tube(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                lambda img: image_transform_optical_flow(
                    img, call_get_flow_from_images(data, data.images.previous, img, optical_flow),
                    step.step_data.redo_flow_factor))


# Conditional Tubes (can be switched on or off by providing a Callable[Boolean] `is_do_process` predicate).
def conditional_hybrid_video_after_generation_tube(step: KeyStep) -> ImageTube:
    data = step.render_data
    step_data = step.step_data
    return tube(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                lambda img: call_hybrid_composite(data, data.indexes.frame.i, img, step_data.hybrid_comp_schedules),
                lambda img: Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                is_do_process=
                lambda: data.indexes.is_not_first_frame() and data.is_hybrid_composite_after_generation())


def conditional_extra_color_match_tube(data: RenderData) -> ImageTube:
    # color matching on first frame is after generation, color match was collected earlier,
    # so we do an extra generation to avoid the corruption introduced by the color match of first output
    return tube(lambda img: maintain_colors(img, data.images.color_match, data.args.anim_args.color_coherence),
                lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                lambda img: maintain_colors(img, data.images.color_match, data.args.anim_args.color_coherence),
                lambda img: Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                is_do_process=
                lambda: data.indexes.is_first_frame() and data.is_initialize_color_match(data.images.color_match))


def conditional_color_match_tube(step: KeyStep) -> ImageTube:
    # on strength 0, set color match to generation
    return tube(lambda img: cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR),
                is_do_process=lambda: step.render_data.is_do_color_match_conversion(step))


def conditional_force_to_grayscale_tube(data: RenderData) -> ImageTube:
    return tube(lambda img: ImageOps.grayscale(img),
                lambda img: ImageOps.colorize(img, black="black", white="white"),
                is_do_process=lambda: data.args.anim_args.color_force_grayscale)


def conditional_add_overlay_mask_tube(data: RenderData, is_tween) -> ImageTube:
    is_use_overlay = data.args.args.overlay_mask
    is_use_mask = data.args.anim_args.use_mask_video or data.args.args.use_mask
    index = data.indexes.tween.i if is_tween else data.indexes.frame.i
    is_bgr_array = True
    return tube(lambda img: ImageOps.grayscale(img),
                lambda img: do_overlay_mask(data.args.args, data.args.anim_args, img, index, is_bgr_array),
                is_do_process=lambda: is_use_overlay and is_use_mask)


def conditional_force_tween_to_grayscale_tube(data: RenderData) -> ImageTube:
    return tube(lambda img: cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY),
                lambda img: cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                is_do_process=lambda: data.args.anim_args.color_force_grayscale)


# Composite Tubes, made from other Tubes.
def contrasted_noise_transformation_tube(data: RenderData, step: KeyStep) -> ImageTube:
    """Combines contrast and noise transformation tubes."""
    contrast_tube: Tube = contrast_transformation_tube(data, step)
    noise_tube: Tube = noise_transformation_tube(data, step)
    return tube(lambda img: noise_tube(contrast_tube(img)))


def conditional_frame_transformation_tube(step: KeyStep, is_tween: bool = False) -> ImageTube:
    hybrid_tube: Tube = conditional_hybrid_video_after_generation_tube(step)
    extra_tube: Tube = conditional_extra_color_match_tube(step.render_data)
    gray_tube: Tube = conditional_force_to_grayscale_tube(step.render_data)
    mask_tube: Tube = conditional_add_overlay_mask_tube(step.render_data, is_tween)
    return tube(lambda img: mask_tube(gray_tube(extra_tube(hybrid_tube(img)))))
