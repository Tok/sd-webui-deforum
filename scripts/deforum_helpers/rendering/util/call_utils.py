from ...animation import anim_frame_warp
from ...generate import generate
from ...hybrid_video import (hybrid_composite, get_matrix_for_hybrid_motion, get_matrix_for_hybrid_motion_prev,
                             get_flow_for_hybrid_motion, get_flow_for_hybrid_motion_prev, image_transform_ransac,
                             image_transform_optical_flow, get_flow_from_images, abs_flow_to_rel_flow,
                             rel_flow_to_abs_flow)
from ...video_audio_utilities import get_frame_name, get_next_frame, render_preview


# Purpose:
# This module mostly exists for refactoring and reducing the complexity of render.py without touching any other modules.
# Currently useful to reduce complexity in calls with many arguments, that have been or will be regrouped into "init".
# Alternatively all methods may be overloaded in the original modules, but doing so would propagate the Init class to
# namespaces where it doesn't really belong, which is rather undesirable.
#
# Form:
# The following functions shouldn't contain any logic and directly return with the call to the actual method.
# - Naming starts with "call_".
# - "init" to be passed as 1st argument.
# - pass frame_idx or twin_frame_idx or other indices as 2nd argument "i" where applicable.

# Animation:
def call_anim_frame_warp(init, i, image, depth):
    return anim_frame_warp(image,
                           init.args.args,
                           init.args.anim_args,
                           init.animation_keys.deform_keys,
                           i,
                           init.depth_model,
                           depth=depth,
                           device=init.args.root.device,
                           half_precision=init.args.root.half_precision)


# Generation:
def call_generate(init, i, schedule):
    return generate(init.args.args,
                    init.animation_keys.deform_keys,
                    init.args.anim_args,
                    init.args.loop_args,
                    init.args.controlnet_args,
                    init.args.root,
                    init.parseq_adapter,
                    i, sampler_name=schedule.sampler_name)


# Hybrid Video
def call_get_flow_from_images(init, prev_image, next_image, cadence):
    # cadence is currently either "optical_flow_redo_generation" or "init.args.anim_args.optical_flow_cadence"
    # TODO try to init "optical_flow_redo_generation" early, then remove the "cadence" arg again
    return get_flow_from_images(prev_image, next_image, cadence, init.animation_mode.raft_model)


def call_hybrid_composite(init, i, image, hybrid_comp_schedules):
    return hybrid_composite(init.args.args,
                            init.args.anim_args,
                            i, image,
                            init.depth_model,
                            hybrid_comp_schedules,
                            init.args.root)


# Video & Audio
def call_render_preview(init, i, last_preview_frame):
    return render_preview(init.args.args,
                          init.args.anim_args,
                          init.args.video_args,
                          init.args.root,
                          i, last_preview_frame)
