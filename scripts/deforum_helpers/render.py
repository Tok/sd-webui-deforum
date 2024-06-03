# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import gc
import os
import random
import time

import PIL
import cv2
import numpy as np
from PIL import Image, ImageOps
from deforum_api import JobStatusTracker
from modules import lowvram, devices, sd_hijack
from modules.shared import opts, cmd_opts, state, sd_model

from .colors import maintain_colors
from .composable_masks import compose_mask_with_check
from .hybrid_video import (get_matrix_for_hybrid_motion, get_matrix_for_hybrid_motion_prev,
                           get_flow_for_hybrid_motion, get_flow_for_hybrid_motion_prev, image_transform_ransac,
                           image_transform_optical_flow, abs_flow_to_rel_flow,
                           rel_flow_to_abs_flow)
from .image_sharpening import unsharp_mask
from .load_images import get_mask, load_img, load_image, get_mask_from_file
from .masks import do_overlay_mask
from .noise import add_noise
from .prompt import prepare_prompt
from .rendering.data.schedule import Schedule
from .rendering.initialization import RenderInit
from .rendering.util import put_if_present, call_anim_frame_warp
from .rendering.util.call_utils import call_get_flow_from_images, call_generate, call_render_preview, \
    call_hybrid_composite
from .rendering.util.memory_utils import MemoryUtils
from .resume import get_resume_vars
from .save_images import save_image
from .seed import next_seed
from .subtitle_handler import write_frame_subtitle, format_animation_params
from .video_audio_utilities import get_frame_name, get_next_frame


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    render_init = RenderInit.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    # TODO method is temporarily torn apart to remove args from direct access in larger execution scope.
    run_render_animation(render_init)


def run_render_animation(init):
    # TODO refactor to try and avoid all usage and reassignments to "init.args.args".
    # TODO try to avoid late init of "prev_flow" or isolate it together with all other moving parts.
    # TODO isolate "depth" with other moving parts
    # TODO cleanup and find potential side effects in 31 refs to init.args.root
    # TODO isolate the relevant data in +250 refs to init.args,
    #  move that stuff to init and eventually try to drop init.args
    #  see dimensions() in RenderInit for an example of delegating the relevant stuff from args.

    # state for interpolating between diffusion steps
    turbo_steps = 1 if init.has_video_input() else init.cadence()
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    # initialize vars
    prev_img = None
    color_match_sample = None
    start_frame = 0

    # resume animation (requires at least two frames - see function)
    if init.is_resuming_from_timestring():
        # determine last frame and frame to start on
        prev_frame, next_frame, prev_img, next_img = get_resume_vars(
            folder=init.args.args.outdir,
            timestring=init.args.anim_args.resume_timestring,
            cadence=turbo_steps)

        # set up turbo step vars
        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = prev_img, prev_frame
            turbo_next_image, turbo_next_frame_idx = next_img, next_frame

        # advance start_frame to next frame
        start_frame = next_frame + 1

    frame_idx = start_frame

    # reset the mask vals as they are overwritten in the compose_mask algorithm
    mask_vals = {}
    noise_mask_vals = {}

    mask_vals['everywhere'] = Image.new('1', init.dimensions(), 1)
    noise_mask_vals['everywhere'] = Image.new('1', init.dimensions(), 1)

    mask_image = None

    if (init.args.args.use_init
            and ((init.args.args.init_image != None
            and init.args.args.init_image != '')
            or init.args.args.init_image_box != None)):
        _, mask_image = load_img(init.args.args.init_image,
                                 init.args.args.init_image_box,
                                 shape=init.dimensions(),
                                 use_alpha_as_mask=init.args.args.use_alpha_as_mask)
        mask_vals['video_mask'] = mask_image
        noise_mask_vals['video_mask'] = mask_image

    # Grab the first frame masks since they wont be provided until next frame
    # Video mask overrides the init image mask, also, won't be searching for init_mask if use_mask_video is set
    # Made to solve https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/386
    if init.args.anim_args.use_mask_video:
        init.args.args.mask_file = get_mask_from_file(
            get_next_frame(init.output_directory, init.args.anim_args.video_mask_path, frame_idx, True),
            init.args.args)
        init.args.root.noise_mask = get_mask_from_file(
            get_next_frame(init.output_directory, init.args.anim_args.video_mask_path, frame_idx, True),
            init.args.args)
        mask_vals['video_mask'] = get_mask_from_file(
            get_next_frame(init.output_directory, init.args.anim_args.video_mask_path, frame_idx, True),
            init.args.args)
        noise_mask_vals['video_mask'] = get_mask_from_file(
            get_next_frame(init.output_directory, init.args.anim_args.video_mask_path, frame_idx, True),
            init.args.args)
    elif mask_image is None and init.is_use_mask:
        mask_vals['video_mask'] = get_mask(init.args.args)
        noise_mask_vals['video_mask'] = get_mask(init.args.args)  # TODO?: add a different default noisc mask

    # get color match for 'Image' color coherence only once, before loop
    if init.args.anim_args.color_coherence == 'Image':
        color_match_sample = load_image(init.args.anim_args.color_coherence_image_path, None)
        color_match_sample = color_match_sample.resize(init.dimensions(), PIL.Image.LANCZOS)
        color_match_sample = cv2.cvtColor(np.array(color_match_sample), cv2.COLOR_RGB2BGR)

    # Webui
    state.job_count = init.args.anim_args.max_frames
    last_preview_frame = 0

    # TODO create a Step class in rendering.data with all the iteration specific info,
    #  then eventually try to replace this while loop with functions that:
    #  - 1. Create a collection of Steps with all the required info that is already known or can be calculated
    #       before we enter the iteration.
    #  - 2. Transform and reprocess the steps however needed (i.e. space out or reassign turbo frames etc.)
    #       TODO cadence framing and logic that is currently working off-index may eventually be moved into a 2nd pass.
    #  - 3. Actually do the render by foreaching over the steps in sequence
    while frame_idx < init.args.anim_args.max_frames:
        # Webui

        state.job = f"frame {frame_idx + 1}/{init.args.anim_args.max_frames}"
        state.job_no = frame_idx + 1

        if state.skipped:
            print("\n** PAUSED **")
            state.skipped = False
            while not state.skipped:
                time.sleep(0.1)
            print("** RESUMING **")

        print(f"\033[36mAnimation frame: \033[0m{frame_idx}/{init.args.anim_args.max_frames}  ")

        # TODO move this to the new key collection
        noise = init.animation_keys.deform_keys.noise_schedule_series[frame_idx]
        strength = init.animation_keys.deform_keys.strength_schedule_series[frame_idx]
        scale = init.animation_keys.deform_keys.cfg_scale_schedule_series[frame_idx]
        contrast = init.animation_keys.deform_keys.contrast_schedule_series[frame_idx]
        kernel = int(init.animation_keys.deform_keys.kernel_schedule_series[frame_idx])
        sigma = init.animation_keys.deform_keys.sigma_schedule_series[frame_idx]
        amount = init.animation_keys.deform_keys.amount_schedule_series[frame_idx]
        threshold = init.animation_keys.deform_keys.threshold_schedule_series[frame_idx]
        cadence_flow_factor = init.animation_keys.deform_keys.cadence_flow_factor_schedule_series[frame_idx]
        redo_flow_factor = init.animation_keys.deform_keys.redo_flow_factor_schedule_series[frame_idx]
        hybrid_comp_schedules = {
            "alpha": init.animation_keys.deform_keys.hybrid_comp_alpha_schedule_series[frame_idx],
            "mask_blend_alpha": init.animation_keys.deform_keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
            "mask_contrast": init.animation_keys.deform_keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_low": int(
                init.animation_keys.deform_keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
            "mask_auto_contrast_cutoff_high": int(
                init.animation_keys.deform_keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
            "flow_factor": init.animation_keys.deform_keys.hybrid_flow_factor_schedule_series[frame_idx]
        }

        # TODO eventually move schedule into new Step class
        schedule = Schedule.create(init.animation_keys.deform_keys, frame_idx, init.args.anim_args, init.args.args)

        if init.is_use_mask and not init.args.anim_args.use_noise_mask:
            noise_mask_seq = schedule.mask_seq

        depth = None

        if init.is_3d_with_med_or_low_vram():
            # Unload the main checkpoint and load the depth model
            lowvram.send_everything_to_cpu()
            sd_hijack.model_hijack.undo_hijack(sd_model)
            devices.torch_gc()
            if init.animation_mode.is_predicting_depths:
                init.animation_mode.depth_model.to(init.args.root.device)

        if turbo_steps == 1 and opts.data.get("deforum_save_gen_info_as_srt"):
            params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
            params_string = format_animation_params(init.animation_keys.deform_keys, init.prompt_series, frame_idx,
                                                    params_to_print)
            write_frame_subtitle(init.srt.filename, frame_idx, init.srt.frame_duration,
                                 f"F#: {frame_idx}; Cadence: false; Seed: {init.args.args.seed}; {params_string}")
            params_string = None  # FIXME ??

        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(start_frame, frame_idx - turbo_steps)
            cadence_flow = None
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                # update progress during cadence
                state.job = f"frame {tween_frame_idx + 1}/{init.args.anim_args.max_frames}"
                state.job_no = tween_frame_idx + 1
                # cadence vars
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx

                # optical flow cadence setup before animation warping
                if (init.args.anim_args.animation_mode in ['2D', '3D']
                        and init.args.anim_args.optical_flow_cadence != 'None'):
                    if init.animation_keys.deform_keys.strength_schedule_series[tween_frame_start_idx] > 0:
                        if cadence_flow is None and turbo_prev_image is not None and turbo_next_image is not None:
                            cadence_flow = call_get_flow_from_images(init, turbo_prev_image, turbo_next_image,
                                                                     init.args.anim_args.optical_flow_cadence) / 2
                            turbo_next_image = image_transform_optical_flow(turbo_next_image, -cadence_flow, 1)

                if opts.data.get("deforum_save_gen_info_as_srt"):
                    params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
                    params_string = format_animation_params(init.animation_keys.deform_keys,
                                                            init.prompt_series,
                                                            tween_frame_idx,
                                                            params_to_print)
                    write_frame_subtitle(init.srt.filename, tween_frame_idx, init.srt.frame_duration,
                                         f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {init.args.args.seed}; {params_string}")
                    params_string = None

                print(
                    f"Creating in-between {'' if cadence_flow is None else init.args.anim_args.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")

                if init.depth_model is not None:
                    assert (turbo_next_image is not None)
                    depth = init.depth_model.predict(turbo_next_image, init.args.anim_args.midas_weight,
                                                     init.args.root.half_precision)

                if advance_prev:
                    turbo_prev_image, _ = call_anim_frame_warp(init, tween_frame_idx, turbo_prev_image, depth)
                if advance_next:
                    turbo_prev_image, _ = call_anim_frame_warp(init, tween_frame_idx, turbo_next_image, depth)

                # hybrid video motion - warps turbo_prev_image or turbo_next_image to match motion
                if tween_frame_idx > 0:
                    if init.args.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                        if init.args.anim_args.hybrid_motion_use_prev_img:
                            matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx - 1,
                                                                       init.dimensions(),
                                                                       init.animation_mode.hybrid_input_files, prev_img,
                                                                       init.args.anim_args.hybrid_motion)
                            if advance_prev:
                                turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                            if advance_next:
                                turbo_next_image = image_transform_ransac(turbo_next_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                        else:
                            matrix = get_matrix_for_hybrid_motion(tween_frame_idx - 1,
                                                                  init.dimensions(),
                                                                  init.animation_mode.hybrid_input_files,
                                                                  init.args.anim_args.hybrid_motion)
                            if advance_prev:
                                turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                            if advance_next:
                                turbo_next_image = image_transform_ransac(turbo_next_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                    if init.args.anim_args.hybrid_motion in ['Optical Flow']:
                        if init.args.anim_args.hybrid_motion_use_prev_img:
                            flow = get_flow_for_hybrid_motion_prev(tween_frame_idx - 1,
                                                                   init.dimensions(),
                                                                   init.animation_mode.hybrid_input_files,
                                                                   init.animation_mode.hybrid_frame_path,
                                                                   init.animation_mode.prev_flow,
                                                                   prev_img,
                                                                   init.args.anim_args.hybrid_flow_method,
                                                                   init.animation_mode.raft_model,
                                                                   init.args.anim_args.hybrid_flow_consistency,
                                                                   init.args.anim_args.hybrid_consistency_blur,
                                                                   init.args.anim_args.hybrid_comp_save_extra_frames)
                            if advance_prev:
                                turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow,
                                                                                hybrid_comp_schedules['flow_factor'])
                            if advance_next:
                                turbo_next_image = image_transform_optical_flow(turbo_next_image, flow,
                                                                                hybrid_comp_schedules['flow_factor'])
                            init.animation_mode.prev_flow = flow  # FIXME shouldn't
                        else:
                            flow = get_flow_for_hybrid_motion(tween_frame_idx - 1, init.dimensions(),
                                                              init.animation_mode.hybrid_input_files,
                                                              init.animation_mode.hybrid_frame_path,
                                                              init.animation_mode.prev_flow,
                                                              init.args.anim_args.hybrid_flow_method,
                                                              init.animation_mode.raft_model,
                                                              init.args.anim_args.hybrid_flow_consistency,
                                                              init.args.anim_args.hybrid_consistency_blur,
                                                              init.args.anim_args.hybrid_comp_save_extra_frames)
                            if advance_prev:
                                turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow,
                                                                                hybrid_comp_schedules['flow_factor'])
                            if advance_next:
                                turbo_next_image = image_transform_optical_flow(turbo_next_image, flow,
                                                                                hybrid_comp_schedules['flow_factor'])
                            init.animation_mode.prev_flow = flow

                # TODO cadence related transforms to be decoupled and handled in a 2nd pass
                # do optical flow cadence after animation warping
                if cadence_flow is not None:
                    cadence_flow = abs_flow_to_rel_flow(cadence_flow, init.width(), init.height())
                    cadence_flow, _ = call_anim_frame_warp(init, tween_frame_idx, cadence_flow, depth)
                    cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, init.width(), init.height()) * tween
                    if advance_prev:
                        turbo_prev_image = image_transform_optical_flow(turbo_prev_image,
                                                                        cadence_flow_inc,
                                                                        cadence_flow_factor)
                    if advance_next:
                        turbo_next_image = image_transform_optical_flow(turbo_next_image,
                                                                        cadence_flow_inc,
                                                                        cadence_flow_factor)

                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
                else:
                    img = turbo_next_image

                # intercept and override to grayscale
                if init.args.anim_args.color_force_grayscale:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # overlay mask
                if init.args.args.overlay_mask and (init.args.anim_args.use_mask_video or init.args.args.use_mask):
                    img = do_overlay_mask(init.args.args, init.args.anim_args, img, tween_frame_idx, True)

                # get prev_img during cadence
                prev_img = img

                # current image update for cadence frames (left commented because it doesn't currently update the preview)
                # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                # saving cadence frames
                filename = f"{init.args.root.timestring}_{tween_frame_idx:09}.png"
                save_path = os.path.join(init.args.args.outdir, filename)
                cv2.imwrite(save_path, img)

                if init.args.anim_args.save_depth_maps:
                    dm_save_path = os.path.join(init.output_directory, f"{init.args.root.timestring}_depth_{tween_frame_idx:09}.png")
                    init.depth_model.save(dm_save_path, depth)

        # get color match for video outside of prev_img conditional
        if init.args.anim_args.color_coherence == 'Video Input' and init.is_hybrid_available():
            if int(frame_idx) % int(init.args.anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(os.path.join(init.output_directory, 'inputframes', get_frame_name(
                    init.args.anim_args.video_init_path) + f"{frame_idx:09}.jpg"))
                prev_vid_img = prev_vid_img.resize(init.dimensions(), PIL.Image.LANCZOS)
                color_match_sample = np.asarray(prev_vid_img)
                color_match_sample = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2BGR)

        # after 1st frame, prev_img exists
        if prev_img is not None:
            # apply transforms to previous frame
            prev_img, depth = call_anim_frame_warp(init, frame_idx, prev_img, None)

            # do hybrid compositing before motion
            if init.is_hybrid_composite_before_motion():
                # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
                _, prev_img = call_hybrid_composite(init, frame_idx, prev_img, hybrid_comp_schedules)

            # hybrid video motion - warps prev_img to match motion, usually to prepare for compositing
            if init.args.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                if init.args.anim_args.hybrid_motion_use_prev_img:
                    matrix = get_matrix_for_hybrid_motion_prev(frame_idx - 1,
                                                               init.dimensions(),
                                                               init.animation_mode.hybrid_input_files, prev_img,
                                                               init.args.anim_args.hybrid_motion)
                else:
                    matrix = get_matrix_for_hybrid_motion(frame_idx - 1,
                                                          init.dimensions(),
                                                          init.animation_mode.hybrid_input_files,
                                                          init.args.anim_args.hybrid_motion)
                prev_img = image_transform_ransac(prev_img, matrix, init.args.anim_args.hybrid_motion)
            if init.args.anim_args.hybrid_motion in ['Optical Flow']:
                if init.args.anim_args.hybrid_motion_use_prev_img:
                    flow = get_flow_for_hybrid_motion_prev(frame_idx - 1,
                                                           init.dimensions(),
                                                           init.animation_mode.hybrid_input_files,
                                                           init.animation_mode.hybrid_frame_path,
                                                           init.animation_mode.prev_flow, prev_img,
                                                           init.args.anim_args.hybrid_flow_method,
                                                           init.animation_mode.raft_model,
                                                           init.args.anim_args.hybrid_flow_consistency,
                                                           init.args.anim_args.hybrid_consistency_blur,
                                                           init.args.anim_args.hybrid_comp_save_extra_frames)
                else:
                    flow = get_flow_for_hybrid_motion(frame_idx - 1,
                                                      init.dimensions(),
                                                      init.animation_mode.hybrid_input_files,
                                                      init.animation_mode.hybrid_frame_path,
                                                      init.animation_mode.prev_flow,
                                                      init.args.anim_args.hybrid_flow_method,
                                                      init.animation_mode.raft_model,
                                                      init.args.anim_args.hybrid_flow_consistency,
                                                      init.args.anim_args.hybrid_consistency_blur,
                                                      init.args.anim_args.hybrid_comp_save_extra_frames)
                prev_img = image_transform_optical_flow(prev_img, flow, hybrid_comp_schedules['flow_factor'])
                init.animation_mode.prev_flow = flow

            # do hybrid compositing after motion (normal)
            if init.is_normal_hybrid_composite():
                # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
                _, prev_img = call_hybrid_composite(init, frame_idx, prev_img, hybrid_comp_schedules)
            # apply color matching
            if init.has_color_coherence():
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, init.args.anim_args.color_coherence)

            # intercept and override to grayscale
            if init.args.anim_args.color_force_grayscale:
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)

            # apply scaling
            contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # anti-blur
            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
                                              mask_image if init.args.args.use_mask else None)
            # apply frame noising
            if init.args.args.use_mask or init.args.anim_args.use_noise_mask:
                init.args.root.noise_mask = compose_mask_with_check(init.args.root,
                                                                    init.args.args,
                                                                    noise_mask_seq, # FIXME might be ref'd b4 assignment
                                                                    noise_mask_vals,
                                                                    Image.fromarray(cv2.cvtColor(contrast_image,
                                                                                                 cv2.COLOR_BGR2RGB)))
            noised_image = add_noise(contrast_image, noise, init.args.args.seed, init.args.anim_args.noise_type,
                                     (init.args.anim_args.perlin_w, init.args.anim_args.perlin_h,
                                      init.args.anim_args.perlin_octaves,
                                      init.args.anim_args.perlin_persistence),
                                     init.args.root.noise_mask, init.args.args.invert_mask)

            # use transformed previous frame as init for current
            init.args.args.use_init = True
            init.args.root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
            init.args.args.strength = max(0.0, min(1.0, strength))

        init.args.args.scale = scale

        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        init.args.args.pix2pix_img_cfg_scale = float(
            init.animation_keys.deform_keys.pix2pix_img_cfg_scale_series[frame_idx])

        # grab prompt for current frame
        init.args.args.prompt = init.prompt_series[frame_idx]

        if init.args.args.seed_behavior == 'schedule' or init.parseq_adapter.manages_seed():
            init.args.args.seed = int(init.animation_keys.deform_keys.seed_schedule_series[frame_idx])

        if init.args.anim_args.enable_checkpoint_scheduling:
            init.args.args.checkpoint = init.animation_keys.deform_keys.checkpoint_schedule_series[frame_idx]
        else:
            init.args.args.checkpoint = None

        # SubSeed scheduling
        if init.args.anim_args.enable_subseed_scheduling:
            init.args.root.subseed = int(init.animation_keys.deform_keys.subseed_schedule_series[frame_idx])
            init.args.root.subseed_strength = float(
                init.animation_keys.deform_keys.subseed_strength_schedule_series[frame_idx])

        if init.parseq_adapter.manages_seed():
            init.args.anim_args.enable_subseed_scheduling = True
            init.args.root.subseed = int(init.animation_keys.deform_keys.subseed_schedule_series[frame_idx])
            init.args.root.subseed_strength = init.animation_keys.deform_keys.subseed_strength_schedule_series[
                frame_idx]

        # set value back into the prompt - prepare and report prompt and seed
        init.args.args.prompt = prepare_prompt(init.args.args.prompt, init.args.anim_args.max_frames,
                                               init.args.args.seed, frame_idx)

        # grab init image for current frame
        if init.animation_mode.has_video_input:
            init_frame = get_next_frame(init.output_directory, init.args.anim_args.video_init_path,
                                        frame_idx, False)
            print(f"Using video init frame {init_frame}")
            init.args.args.init_image = init_frame
            init.args.args.init_image_box = None  # init_image_box not used in this case
            init.args.args.strength = max(0.0, min(1.0, strength))
        if init.args.anim_args.use_mask_video:
            mask_init_frame = get_next_frame(init.output_directory, init.args.anim_args.video_mask_path,
                                             frame_idx, True)
            temp_mask = get_mask_from_file(mask_init_frame, init.args.args)
            init.args.args.mask_file = temp_mask
            init.args.root.noise_mask = temp_mask
            mask_vals['video_mask'] = temp_mask

        if init.args.args.use_mask:
            init.args.args.mask_image = compose_mask_with_check(init.args.root, init.args.args,
                                                                schedule.mask_seq, mask_vals,
                                                                init.args.root.init_sample) \
                if init.args.root.init_sample is not None else None  # we need it only after the first frame anyway

        init.animation_keys.update(frame_idx)
        setup_opts(init, schedule)

        if init.is_3d_with_med_or_low_vram():
            if init.animation_mode.is_predicting_depths: init.depth_model.to('cpu')
            devices.torch_gc()
            lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            sd_hijack.model_hijack.hijack(sd_model)

        # TODO try init early, also see "call_get_flow_from_images"
        optical_flow_redo_generation = init.args.anim_args.optical_flow_redo_generation \
            if not init.args.args.motion_preview_mode else 'None'

        # optical flow redo before generation
        if optical_flow_redo_generation != 'None' and prev_img is not None and strength > 0:
            stored_seed = init.args.args.seed
            init.args.args.seed = random.randint(0, 2 ** 32 - 1)
            print(
                f"Optical flow redo is diffusing and warping using {optical_flow_redo_generation} and seed {init.args.args.seed} optical flow before generation.")

            disposable_image = call_generate(init, frame_idx, schedule)
            disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
            disposable_flow = call_get_flow_from_images(init, prev_img, disposable_image,
                                                        optical_flow_redo_generation) / 2
            disposable_image = cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB)
            disposable_image = image_transform_optical_flow(disposable_image, disposable_flow, redo_flow_factor)
            init.args.args.seed = stored_seed
            init.args.root.init_sample = Image.fromarray(disposable_image)
            del (disposable_image, disposable_flow, stored_seed)
            gc.collect()

        # diffusion redo
        if (int(init.args.anim_args.diffusion_redo) > 0
                and prev_img is not None and strength > 0
                and not init.args.args.motion_preview_mode):
            stored_seed = init.args.args.seed
            for n in range(0, int(init.args.anim_args.diffusion_redo)):
                print(f"Redo generation {n + 1} of {int(init.args.anim_args.diffusion_redo)} before final generation")
                init.args.args.seed = random.randint(0, 2 ** 32 - 1)
                disposable_image = call_generate(init, frame_idx, schedule)
                disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
                # color match on last one only
                if n == int(init.args.anim_args.diffusion_redo):
                    disposable_image = maintain_colors(prev_img, color_match_sample,
                                                       init.args.anim_args.color_coherence)
                init.args.args.seed = stored_seed
                init.args.root.init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
            del (disposable_image, stored_seed)  # FIXME disposable_image might be referenced before assignment.
            gc.collect()  # TODO try to eventually kick the gc only once at the end of every generation or iteration.

        # generation
        image = call_generate(init, frame_idx, schedule)

        if image is None:
            break

        # do hybrid video after generation
        if frame_idx > 0 and init.is_hybrid_composite_after_generation():
            temp_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
            _, temp_image_2 = call_hybrid_composite(init, frame_idx, temp_image, hybrid_comp_schedules)
            image = Image.fromarray(cv2.cvtColor(temp_image_2, cv2.COLOR_BGR2RGB))

        # color matching on first frame is after generation, color match was collected earlier,
        # so we do an extra generation to avoid the corruption introduced by the color match of first output
        if frame_idx == 0 and init.is_color_match_to_be_initialized(color_match_sample):
            temp_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            temp_image = maintain_colors(temp_color, color_match_sample, init.args.anim_args.color_coherence)
            image = Image.fromarray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))

        # intercept and override to grayscale
        if init.args.anim_args.color_force_grayscale:
            image = ImageOps.grayscale(image)
            image = ImageOps.colorize(image, black="black", white="white")

        # overlay mask
        if init.args.args.overlay_mask and (init.args.anim_args.use_mask_video or init.is_use_mask):
            image = do_overlay_mask(init.args.args, init.args.anim_args, image, frame_idx)

        # on strength 0, set color match to generation
        if (((not init.args.anim_args.legacy_colormatch and not init.args.args.use_init)
             or (init.args.anim_args.legacy_colormatch and strength == 0))
                and init.args.anim_args.color_coherence not in ['Image', 'Video Input']):
            color_match_sample = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not init.animation_mode.has_video_input:
            prev_img = opencv_image

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = opencv_image, frame_idx
            frame_idx += turbo_steps
        else:
            filename = f"{init.args.root.timestring}_{frame_idx:09}.png"
            save_image(image, 'PIL', filename, init.args.args, init.args.video_args, init.args.root)

            if init.args.anim_args.save_depth_maps:
                # TODO move all depth related stuff to new class. (also see RenderInit)
                if MemoryUtils.is_low_or_med_vram():
                    lowvram.send_everything_to_cpu()
                    sd_hijack.model_hijack.undo_hijack(sd_model)
                    devices.torch_gc()
                    init.depth_model.to(init.args.root.device)
                depth = init.depth_model.predict(opencv_image, init.args.anim_args.midas_weight,
                                                 init.args.root.half_precision)
                init.depth_model.save(
                    os.path.join(init.output_directory, f"{init.args.root.timestring}_depth_{frame_idx:09}.png"), depth)
                if MemoryUtils.is_low_or_med_vram():
                    init.depth_model.to('cpu')
                    devices.torch_gc()
                    lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
                    sd_hijack.model_hijack.hijack(sd_model)
            frame_idx += 1

        last_preview_frame = progress_and_make_preview(init, image, frame_idx, state, last_preview_frame)
        update_tracker(init.args.root, frame_idx, init.args.anim_args)
        init.animation_mode.cleanup()


def setup_opts(init, schedule):
    if init.has_img2img_fix_steps():
        # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
        init.opts.data["img2img_fix_steps"] = False
    put_if_present(init.args.opts.data, "CLIP_stop_at_last_layers", schedule.clipskip)
    put_if_present(init.args.opts.data, "initial_noise_multiplier", schedule.noise_multiplier)
    put_if_present(init.args.opts.data, "eta_ddim", schedule.eta_ddim)
    put_if_present(init.args.opts.data, "eta_ancestral", schedule.eta_ancestral)


def progress_and_make_preview(init, image, frame_idx, state, last_preview_frame):
    state.assign_current_image(image)
    init.args.args.seed = next_seed(init.args.args, init.args.root)  # TODO refactor assignment
    return call_render_preview(init, frame_idx, last_preview_frame)


def update_tracker(root, frame_idx, anim_args):
    JobStatusTracker().update_phase(root.job_id, phase="GENERATING", progress=frame_idx / anim_args.max_frames)
