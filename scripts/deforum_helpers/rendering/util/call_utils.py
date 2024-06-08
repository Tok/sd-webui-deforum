from .utils import context
from ...animation import anim_frame_warp
from ...generate import generate
from ...hybrid_video import (
    # Functions related to flow calculation
    get_flow_from_images,
    get_flow_for_hybrid_motion,
    get_flow_for_hybrid_motion_prev,

    # Functions related to matrix calculation
    get_matrix_for_hybrid_motion,
    get_matrix_for_hybrid_motion_prev,

    # Other hybrid functions
    hybrid_composite)
from ...load_images import get_mask_from_file
from ...subtitle_handler import format_animation_params, write_frame_subtitle
from ...video_audio_utilities import get_next_frame, render_preview

"""
This module provides utility functions for simplifying calls to other modules within the `render.py` module.

**Purpose:**
- **Reduce Argument Complexity:**  Provides a way to call functions in other modules without directly handling 
  a large number of complex arguments. This simplifies code within `render.py` by encapsulating argument management.
- **Minimize Namespace Pollution:**  Provides an alternative to overloading methods in the original modules, 
  which would introduce the `RenderInit` class into namespaces where it's not inherently needed.

**Structure:**
- **Simple Call Forwarding:** Functions in this module primarily act as wrappers. They perform minimal logic, 
  often just formatting or passing arguments, and directly call the corresponding method.
- **Naming Convention:**
    - Function names begin with "call_", followed by the name of the actual method to call.
    - The `init` object is always passed as the first argument.
    - Frame indices (e.g., `frame_idx`, `twin_frame_idx`) are passed as the second argument "i", when relevant.

**Example:**
```python
# Example function in this module
def call_some_function(init, i, ...):
    return some_module.some_function(init.arg77, init.arg.arg.whatever, i, ...)
```
"""


# Animation:
def call_anim_frame_warp(init, i, image, depth):
    with context(init.args) as ia:
        return anim_frame_warp(image, ia.args, ia.anim_args, init.animation_keys.deform_keys, i, init.depth_model,
                               depth=depth, device=ia.root.device, half_precision=ia.root.half_precision)


# Generation:
def call_generate(init, i, schedule):
    with context(init.args) as ia:
        return generate(ia.args, init.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args,
                        ia.root, init.parseq_adapter, i, sampler_name=schedule.sampler_name)


# Hybrid Video
def call_get_flow_from_images(init, prev_image, next_image, cadence):
    # cadence is currently either "optical_flow_redo_generation" or "init.args.anim_args.optical_flow_cadence"
    # TODO try to init "optical_flow_redo_generation" early, then remove the "cadence" arg again
    return get_flow_from_images(prev_image, next_image, cadence, init.animation_mode.raft_model)


def call_get_flow_for_hybrid_motion_prev(init, i, image):
    with context(init.animation_mode) as mode:
        with context(init.args.anim_args) as aa:
            return get_flow_for_hybrid_motion_prev(i, init.dimensions(),
                                                   mode.hybrid_input_files,
                                                   mode.hybrid_frame_path,
                                                   mode.prev_flow,
                                                   image,
                                                   aa.hybrid_flow_method,
                                                   mode.raft_model,
                                                   aa.hybrid_flow_consistency,
                                                   aa.hybrid_consistency_blur,
                                                   aa.hybrid_comp_save_extra_frames)


def call_get_flow_for_hybrid_motion(init, i):
    with context(init.animation_mode) as mode:
        with context(init.args.anim_args) as args:
            return get_flow_for_hybrid_motion(i, init.dimensions(), mode.hybrid_input_files, mode.hybrid_frame_path,
                                              mode.prev_flow, args.hybrid_flow_method, mode.raft_model,
                                              args.hybrid_flow_consistency, args.hybrid_consistency_blur, args)


def call_get_matrix_for_hybrid_motion_prev(init, i, image):
    return get_matrix_for_hybrid_motion_prev(i, init.dimensions(), init.animation_mode.hybrid_input_files,
                                             image, init.args.anim_args.hybrid_motion)


def call_get_matrix_for_hybrid_motion(init, i):
    return get_matrix_for_hybrid_motion(i, init.dimensions(), init.animation_mode.hybrid_input_files,
                                        init.args.anim_args.hybrid_motion)


def call_hybrid_composite(init, i, image, hybrid_comp_schedules):
    with context(init.args) as ia:
        return hybrid_composite(ia.args, ia.anim_args, i, image, init.depth_model,
                                hybrid_comp_schedules, init.args.root)


# Load Images
def call_get_mask_from_file(init, i, is_mask: bool = False):
    next_frame = get_next_frame(init.output_directory, init.args.anim_args.video_mask_path, i, is_mask)
    return get_mask_from_file(next_frame, init.args.args)


def call_get_mask_from_file_with_frame(init, frame):
    return get_mask_from_file(frame, init.args.args)


# Subtitle
def call_format_animation_params(init, i, params_to_print):
    return format_animation_params(init.animation_keys.deform_keys, init.prompt_series, i, params_to_print)


def call_write_frame_subtitle(init, i, params_string, is_cadence: bool = False) -> None:
    text = f"F#: {i}; Cadence: {is_cadence}; Seed: {init.args.args.seed}; {params_string}"
    write_frame_subtitle(init.srt.filename, i, init.srt.frame_duration, text)


# Video & Audio
def call_render_preview(init, i, last_preview_frame):
    with context(init.args) as ia:
        return render_preview(ia.args, ia.anim_args, ia.video_args, ia.root, i, last_preview_frame)


def call_get_next_frame(init, i, video_path, is_mask: bool = False):
    return get_next_frame(init.output_directory, video_path, i, is_mask)
