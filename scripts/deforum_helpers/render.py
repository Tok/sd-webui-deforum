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
from .hybrid_video import (image_transform_ransac, image_transform_optical_flow,
                           abs_flow_to_rel_flow, rel_flow_to_abs_flow)
from .image_sharpening import unsharp_mask
from .masks import do_overlay_mask
from .noise import add_noise
from .prompt import prepare_prompt
from .rendering.data import Turbo
from .rendering.data.schedule import Schedule
from .rendering.data.images import Images
from .rendering.data.indexes import Indexes
from .rendering.data.mask import Mask
from .rendering.initialization import RenderInit, StepInit
from .rendering.util import put_if_present
from .rendering.util.call_utils import (
    # Animation Functions
    call_anim_frame_warp,
    call_format_animation_params,
    call_get_next_frame,
    call_write_frame_subtitle,

    # Generation and Rendering
    call_generate,
    call_render_preview,

    # Hybrid Motion Functions
    call_get_flow_for_hybrid_motion,
    call_get_flow_for_hybrid_motion_prev,
    call_get_matrix_for_hybrid_motion,
    call_get_matrix_for_hybrid_motion_prev,
    call_hybrid_composite,

    # Mask Functions
    call_get_mask_from_file,
    call_get_mask_from_file_with_frame,

    # Flow Functions
    call_get_flow_from_images)
from .rendering.util.memory_utils import MemoryUtils
from .rendering.util.utils import context
from .save_images import save_image
from .seed import next_seed
from .video_audio_utilities import get_frame_name


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    render_init = RenderInit.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    # TODO method is temporarily torn apart to remove args from direct access in larger execution scope.
    run_render_animation(render_init)


def run_render_animation_controlled(init):
    # TODO create a Step class in rendering.data with all the iteration specific info,
    #  then eventually try to replace the main while loop in `run_render_animation` with functions that:
    #  - 1. Create a collection of Steps with all the required info that is already known or can be calculated
    #       before we enter the iteration.
    #  - 2. Transform and reprocess the steps however needed (i.e. space out or reassign turbo frames etc.)
    #       TODO cadence framing and logic that is currently working off-index may eventually be moved into a 2nd pass.
    #  - 3. Actually do the render by foreaching over the steps in sequence
    # TODO also create a SubStep class for the inner for-loop in `run_render_animation` (maybe do that 1st).
    raise NotImplementedError("not implemented.")


def run_render_animation(init):
    # TODO try to avoid late init of "prev_flow" or isolate it together with all other moving parts.
    # TODO isolate "depth" with other moving parts

    images = Images.create(init)
    turbo = Turbo.create(init)  # state for interpolating between diffusion steps
    indexes = Indexes.create(init, turbo)  # TODO reconsider index set...
    mask = Mask.create(init, indexes.frame)  # reset the mask vals as they are overwritten in the compose_mask algorithm

    _init_web_ui_job(init)
    last_preview_frame = 0
    while indexes.frame < init.args.anim_args.max_frames:
        _update_web_ui_job(init, indexes)
        print(f"\033[36mAnimation frame: \033[0m{indexes.frame}/{init.args.anim_args.max_frames}  ")

        # TODO Stuff to be moved into new Step class in `run_render_animation_controlled`:
        step_init = StepInit.create(init.animation_keys.deform_keys, indexes.frame)
        schedule = Schedule.create(init.animation_keys.deform_keys, indexes.frame, init.args.anim_args, init.args.args)

        if init.is_use_mask and not init.args.anim_args.use_noise_mask:
            noise_mask_seq = schedule.mask_seq

        depth = None

        if init.is_3d_with_med_or_low_vram():
            # Unload the main checkpoint and load the depth model
            lowvram.send_everything_to_cpu()
            sd_hijack.model_hijack.undo_hijack(sd_model)
            devices.torch_gc()
            if init.animation_mode.is_predicting_depths:
                init.animation_mode.depth_model.to(init.root.device)

        if turbo.is_first_step_with_subtitles(init):
            params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
            params_string = call_format_animation_params(init, indexes.frame, params_to_print)
            call_write_frame_subtitle(init, indexes.frame, params_string)
            params_string = None  # FIXME ??

        if turbo.is_emit_in_between_frames():
            indexes.tween_frame_start = max(indexes.start_frame, indexes.frame - turbo.steps)
            cadence_flow = None
            # TODO stuff for new SubStep class to be used in `run_render_animation_controlled`
            for indexes.tween_frame in range(indexes.tween_frame_start, indexes.frame):
                # update progress during cadence
                state.job = f"frame {indexes.tween_frame + 1}/{init.args.anim_args.max_frames}"
                state.job_no = indexes.tween_frame + 1
                # cadence vars
                tween = (float(indexes.tween_frame - indexes.tween_frame_start + 1) /
                         float(indexes.frame - indexes.tween_frame_start))

                # optical flow cadence setup before animation warping
                if (init.args.anim_args.animation_mode in ['2D', '3D']
                        and init.args.anim_args.optical_flow_cadence != 'None'):
                    if init.animation_keys.deform_keys.strength_schedule_series[indexes.tween_frame_start] > 0:
                        if cadence_flow is None and turbo.prev_image is not None and turbo.next_image is not None:
                            cadence_flow = call_get_flow_from_images(init, turbo.prev_image, turbo.next_image,
                                                                     init.args.anim_args.optical_flow_cadence) / 2
                            turbo.next_image = image_transform_optical_flow(turbo.next_image, -cadence_flow, 1)

                if opts.data.get("deforum_save_gen_info_as_srt"):
                    params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
                    params_string = call_format_animation_params(init, indexes.tween_frame, params_to_print)
                    call_write_frame_subtitle(init, indexes.tween_frame, params_string, tween < 1.0)
                    params_string = None

                msg_flow_name = '' if cadence_flow is None \
                    else init.args.anim_args.optical_flow_cadence + ' optical flow '
                msg_frame_info = f"cadence frame: {indexes.tween_frame}; tween:{tween:0.2f};"
                print(f"Creating in-between {msg_flow_name}{msg_frame_info}")

                if init.depth_model is not None:
                    assert (turbo.next_image is not None)
                    depth = init.depth_model.predict(turbo.next_image, init.args.anim_args.midas_weight,
                                                     init.root.half_precision)

                # TODO collect images
                if turbo.is_advance_prev(indexes.tween_frame):
                    turbo.prev_image, _ = call_anim_frame_warp(init, indexes.tween_frame, turbo.prev_image, depth)
                if turbo.is_advance_next(indexes.tween_frame):
                    turbo.prev_image, _ = call_anim_frame_warp(init, indexes.tween_frame, turbo.next_image, depth)

                # hybrid video motion - warps turbo.prev_image or turbo.next_image to match motion
                if indexes.tween_frame > 0:
                    if init.args.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                        if init.args.anim_args.hybrid_motion_use_prev_img:
                            matrix = call_get_matrix_for_hybrid_motion_prev(init, indexes.tween_frame - 1, images.previous)
                            if turbo.is_advance_prev(indexes.tween_frame):
                                turbo.prev_image = image_transform_ransac(turbo.prev_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                            if turbo.is_advance_next(indexes.tween_frame):
                                turbo.next_image = image_transform_ransac(turbo.next_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                        else:
                            matrix = call_get_matrix_for_hybrid_motion(init, indexes.tween_frame - 1)
                            if turbo.is_advance_prev(indexes.tween_frame):
                                turbo.prev_image = image_transform_ransac(turbo.prev_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                            if turbo.is_advance_next(indexes.tween_frame):
                                turbo.next_image = image_transform_ransac(turbo.next_image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                    if init.args.anim_args.hybrid_motion in ['Optical Flow']:
                        if init.args.anim_args.hybrid_motion_use_prev_img:
                            flow = call_get_flow_for_hybrid_motion_prev(init, indexes.tween_frame - 1, images.previous)
                            if turbo.is_advance_prev(indexes.tween_frame):
                                turbo.prev_image = image_transform_optical_flow(turbo.prev_image, flow,
                                                                                step_init.flow_factor())
                            if turbo.is_advance_next(indexes.tween_frame):
                                turbo.next_image = image_transform_optical_flow(turbo.next_image, flow,
                                                                                step_init.flow_factor())
                            init.animation_mode.prev_flow = flow
                        else:
                            flow = call_get_flow_for_hybrid_motion(init, indexes.tween_frame - 1)
                            if turbo.is_advance_prev(indexes.tween_frame):
                                turbo.prev_image = image_transform_optical_flow(turbo.prev_image, flow,
                                                                                step_init.flow_factor())
                            if turbo.is_advance_next(indexes.tween_frame):
                                turbo.next_image = image_transform_optical_flow(turbo.next_image, flow,
                                                                                step_init.flow_factor())
                            init.animation_mode.prev_flow = flow

                # TODO cadence related transforms to be decoupled and handled in a 2nd pass
                # do optical flow cadence after animation warping
                with context(cadence_flow) as cf:
                    if cf is not None:
                        cf = abs_flow_to_rel_flow(cf, init.width(), init.height())
                        cf, _ = call_anim_frame_warp(init, indexes.tween_frame, cf, depth)
                        cadence_flow_inc = rel_flow_to_abs_flow(cf, init.width(), init.height()) * tween
                        if turbo.is_advance_prev():
                            turbo.prev_image = image_transform_optical_flow(turbo.prev_image, cadence_flow_inc,
                                                                            step_init.cadence_flow_factor)
                        if turbo.is_advance_next():
                            turbo.next_image = image_transform_optical_flow(turbo.next_image, cadence_flow_inc,
                                                                            step_init.cadence_flow_factor)

                turbo.prev_frame_idx = turbo.next_frame_idx = indexes.tween_frame

                if turbo.prev_image is not None and tween < 1.0:
                    img = turbo.prev_image * (1.0 - tween) + turbo.next_image * tween
                else:
                    img = turbo.next_image

                # intercept and override to grayscale
                if init.args.anim_args.color_force_grayscale:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # overlay mask
                if init.args.args.overlay_mask and (init.args.anim_args.use_mask_video or init.args.args.use_mask):
                    img = do_overlay_mask(init.args.args, init.args.anim_args, img, indexes.tween_frame, True)

                # get images.previous during cadence
                images.previous = img

                # current image update for cadence frames (left commented because it doesn't currently update the preview)
                # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                # saving cadence frames
                filename = f"{init.root.timestring}_{indexes.tween_frame:09}.png"
                save_path = os.path.join(init.args.args.outdir, filename)
                cv2.imwrite(save_path, img)

                if init.args.anim_args.save_depth_maps:
                    dm_save_path = os.path.join(init.output_directory,
                                                f"{init.root.timestring}_depth_{indexes.tween_frame:09}.png")
                    init.depth_model.save(dm_save_path, depth)

        # get color match for video outside of images.previous conditional
        if init.args.anim_args.color_coherence == 'Video Input' and init.is_hybrid_available():
            if int(indexes.frame) % int(init.args.anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(os.path.join(init.output_directory, 'inputframes', get_frame_name(
                    init.args.anim_args.video_init_path) + f"{indexes.frame:09}.jpg"))
                prev_vid_img = prev_vid_img.resize(init.dimensions(), PIL.Image.LANCZOS)
                images.color_match = np.asarray(prev_vid_img)
                images.color_match = cv2.cvtColor(images.color_match, cv2.COLOR_RGB2BGR)

        # after 1st frame, images.previous exists
        if images.previous is not None:
            # apply transforms to previous frame
            images.previous, depth = call_anim_frame_warp(init, indexes.frame, images.previous, None)

            # do hybrid compositing before motion
            if init.is_hybrid_composite_before_motion():
                # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
                _, images.previous = call_hybrid_composite(init, indexes.frame, images.previous, step_init.hybrid_comp_schedules)

            # hybrid video motion - warps images.previous to match motion, usually to prepare for compositing
            with context(init.args.anim_args) as aa:
                if aa.hybrid_motion in ['Affine', 'Perspective']:
                    if aa.hybrid_motion_use_prev_img:
                        matrix = call_get_matrix_for_hybrid_motion_prev(init, indexes.frame - 1, images.previous)
                    else:
                        matrix = call_get_matrix_for_hybrid_motion(init, indexes.frame - 1)
                    images.previous = image_transform_ransac(images.previous, matrix, aa.hybrid_motion)
                if aa.hybrid_motion in ['Optical Flow']:
                    if aa.hybrid_motion_use_prev_img:
                        flow = call_get_flow_for_hybrid_motion_prev(init, indexes.frame - 1, images.previous)
                    else:
                        flow = call_get_flow_for_hybrid_motion(init, indexes.frame - 1)
                    images.previous = image_transform_optical_flow(images.previous, flow, step_init.flow_factor())
                    init.animation_mode.prev_flow = flow

            # do hybrid compositing after motion (normal)
            if init.is_normal_hybrid_composite():
                # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
                _, images.previous = call_hybrid_composite(init, indexes.frame, images.previous, step_init.hybrid_comp_schedules)
            # apply color matching
            if init.has_color_coherence():
                if images.color_match is None:
                    images.color_match = images.previous.copy()
                else:
                    images.previous = maintain_colors(images.previous, images.color_match, init.args.anim_args.color_coherence)

            # intercept and override to grayscale
            if init.args.anim_args.color_force_grayscale:
                images.previous = cv2.cvtColor(images.previous, cv2.COLOR_BGR2GRAY)
                images.previous = cv2.cvtColor(images.previous, cv2.COLOR_GRAY2BGR)

            # apply scaling
            contrast_image = (images.previous * step_init.contrast).round().astype(np.uint8)
            # anti-blur
            if step_init.amount > 0:
                step_init.kernel_size()
                contrast_image = unsharp_mask(contrast_image,
                                              (step_init.kernel, step_init.kernel),
                                              step_init.sigma,
                                              step_init.amount,
                                              step_init.threshold,
                                              mask.image if init.args.args.use_mask else None)
            # apply frame noising
            if init.args.args.use_mask or init.args.anim_args.use_noise_mask:
                init.root.noise_mask = compose_mask_with_check(init.root,
                                                               init.args.args,
                                                               noise_mask_seq,
                                                               # FIXME might be ref'd b4 assignment
                                                               mask.noise_vals,
                                                               Image.fromarray(cv2.cvtColor(contrast_image,
                                                                                            cv2.COLOR_BGR2RGB)))

            with context(init.args.anim_args) as aa:
                noised_image = add_noise(contrast_image, step_init.noise, init.args.args.seed, aa.noise_type,
                                         (aa.perlin_w, aa.perlin_h, aa.perlin_octaves, aa.perlin_persistence),
                                         init.root.noise_mask, init.args.args.invert_mask)

            # use transformed previous frame as init for current
            init.args.args.use_init = True
            init.root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
            init.args.args.strength = max(0.0, min(1.0, step_init.strength))

        init.args.args.scale = step_init.scale

        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        init.args.args.pix2pix_img_cfg_scale = float(
            init.animation_keys.deform_keys.pix2pix_img_cfg_scale_series[indexes.frame])

        # grab prompt for current frame
        init.args.args.prompt = init.prompt_series[indexes.frame]

        with context(init.args) as ia:
            with context(init.animation_keys.deform_keys) as keys:
                # FIXME? check ia.args.seed_behavior
                if ia.args.seed_behavior == 'schedule' or init.parseq_adapter.manages_seed():
                    ia.args.seed = int(keys.seed_schedule_series[indexes.frame])
                if ia.anim_args.enable_checkpoint_scheduling:
                    ia.args.checkpoint = keys.checkpoint_schedule_series[indexes.frame]
                else:
                    ia.args.checkpoint = None

                # SubSeed scheduling
                if ia.anim_args.enable_subseed_scheduling:
                    init.root.subseed = int(keys.subseed_schedule_series[indexes.frame])
                    init.root.subseed_strength = float(keys.subseed_strength_schedule_series[indexes.frame])
                if init.parseq_adapter.manages_seed():
                    init.args.anim_args.enable_subseed_scheduling = True
                    init.root.subseed = int(keys.subseed_schedule_series[indexes.frame])
                    init.root.subseed_strength = keys.subseed_strength_schedule_series[indexes.frame]

            # set value back into the prompt - prepare and report prompt and seed
            ia.args.prompt = prepare_prompt(ia.args.prompt, ia.anim_args.max_frames, ia.args.seed, indexes.frame)
            # grab init image for current frame
            if init.animation_mode.has_video_input:
                init_frame = call_get_next_frame(init, indexes.frame, ia.anim_args.video_init_path)
                print(f"Using video init frame {init_frame}")
                ia.args.init_image = init_frame
                ia.args.init_image_box = None  # init_image_box not used in this case
                ia.args.strength = max(0.0, min(1.0, step_init.strength))
            if ia.anim_args.use_mask_video:
                mask_init_frame = call_get_next_frame(init, indexes.frame, ia.anim_args.video_mask_path, True)
                temp_mask = call_get_mask_from_file_with_frame(init, mask_init_frame)
                ia.args.mask_file = temp_mask
                init.root.noise_mask = temp_mask
                mask.vals['video_mask'] = temp_mask

            if ia.args.use_mask:
                # TODO figure why this is different from mask.image
                ia.args.mask_image = compose_mask_with_check(init.root, ia.args, schedule.mask_seq,
                                                             mask.vals, init.root.init_sample) \
                    if init.root.init_sample is not None else None  # we need it only after the first frame anyway

        init.animation_keys.update(indexes.frame)
        _setup_opts(init, schedule)

        if init.is_3d_with_med_or_low_vram():
            if init.animation_mode.is_predicting_depths: init.depth_model.to('cpu')
            devices.torch_gc()
            lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            sd_hijack.model_hijack.hijack(sd_model)

        # TODO try init early, also see "call_get_flow_from_images"
        optical_flow_redo_generation = init.args.anim_args.optical_flow_redo_generation \
            if not init.args.args.motion_preview_mode else 'None'

        # optical flow redo before generation
        if optical_flow_redo_generation != 'None' and images.previous is not None and step_init.strength > 0:
            stored_seed = init.args.args.seed
            init.args.args.seed = random.randint(0, 2 ** 32 - 1)  # TODO move elsewhere
            msg_start = "Optical flow redo is diffusing and warping using"
            msg_end = "optical flow before generation."
            print(f"{msg_start} {optical_flow_redo_generation} and seed {init.args.args.seed} {msg_end}")

            with context(call_generate(init, indexes.frame, schedule)) as img:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                disposable_flow = call_get_flow_from_images(init, images.previous, img, optical_flow_redo_generation)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = image_transform_optical_flow(img, disposable_flow, step_init.redo_flow_factor)
                init.args.args.seed = stored_seed  # TODO check if (or make) unnecessary and group seeds
                init.root.init_sample = Image.fromarray(img)
                disposable_image = img  # TODO refactor
                del (img, disposable_flow, stored_seed)
                gc.collect()

        # diffusion redo
        if (int(init.args.anim_args.diffusion_redo) > 0
                and images.previous is not None and step_init.strength > 0
                and not init.args.args.motion_preview_mode):
            stored_seed = init.args.args.seed
            for n in range(0, int(init.args.anim_args.diffusion_redo)):
                print(f"Redo generation {n + 1} of {int(init.args.anim_args.diffusion_redo)} before final generation")
                init.args.args.seed = random.randint(0, 2 ** 32 - 1)  # TODO move elsewhere
                disposable_image = call_generate(init, indexes.frame, schedule)
                disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
                # color match on last one only
                if n == int(init.args.anim_args.diffusion_redo):
                    disposable_image = maintain_colors(images.previous, images.color_match,
                                                       init.args.anim_args.color_coherence)
                init.args.args.seed = stored_seed
                init.root.init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
            del (disposable_image, stored_seed)
            gc.collect()  # TODO try to eventually kick the gc only once at the end of every generation, iteration.

        # generation
        image = call_generate(init, indexes.frame, schedule)

        if image is None:
            break

        # do hybrid video after generation
        if indexes.frame > 0 and init.is_hybrid_composite_after_generation():
            temp_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
            _, temp_image_2 = call_hybrid_composite(init, indexes.frame, temp_image, step_init.hybrid_comp_schedules)
            image = Image.fromarray(cv2.cvtColor(temp_image_2, cv2.COLOR_BGR2RGB))

        # color matching on first frame is after generation, color match was collected earlier,
        # so we do an extra generation to avoid the corruption introduced by the color match of first output
        if indexes.frame == 0 and init.is_color_match_to_be_initialized(images.color_match):
            temp_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            temp_image = maintain_colors(temp_color, images.color_match, init.args.anim_args.color_coherence)
            image = Image.fromarray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))

        # intercept and override to grayscale
        if init.args.anim_args.color_force_grayscale:
            image = ImageOps.grayscale(image)
            image = ImageOps.colorize(image, black="black", white="white")

        # overlay mask
        if init.args.args.overlay_mask and (init.args.anim_args.use_mask_video or init.is_use_mask):
            image = do_overlay_mask(init.args.args, init.args.anim_args, image, indexes.frame)

        # on strength 0, set color match to generation
        if (((not init.args.anim_args.legacy_colormatch and not init.args.args.use_init)
             or (init.args.anim_args.legacy_colormatch and step_init.strength == 0))
                and init.args.anim_args.color_coherence not in ['Image', 'Video Input']):
            images.color_match = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not init.animation_mode.has_video_input:
            images.previous = opencv_image

        if turbo.steps > 1:
            turbo.prev_image, turbo.prev_frame_idx = turbo.next_image, turbo.next_frame_idx
            turbo.next_image, turbo.next_frame_idx = opencv_image, indexes.frame
            indexes.frame += turbo.steps
        else:
            filename = f"{init.root.timestring}_{indexes.frame:09}.png"
            save_image(image, 'PIL', filename, init.args.args, init.args.video_args, init.root)

            if init.args.anim_args.save_depth_maps:
                # TODO move all depth related stuff to new class. (also see RenderInit)
                if MemoryUtils.is_low_or_med_vram():
                    lowvram.send_everything_to_cpu()
                    sd_hijack.model_hijack.undo_hijack(sd_model)
                    devices.torch_gc()
                    init.depth_model.to(init.root.device)
                depth = init.depth_model.predict(opencv_image, init.args.anim_args.midas_weight,
                                                 init.root.half_precision)
                init.depth_model.save(
                    os.path.join(init.output_directory, f"{init.root.timestring}_depth_{indexes.frame:09}.png"), depth)
                if MemoryUtils.is_low_or_med_vram():
                    init.depth_model.to('cpu')
                    devices.torch_gc()
                    lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
                    sd_hijack.model_hijack.hijack(sd_model)
            indexes.frame += 1

        # TODO consider state is a global import, so wouldn't really need passing here
        state.assign_current_image(image)
        # may reassign init.args.args and/or root.seed_internal # FIXME?
        init.args.args.seed = next_seed(init.args.args, init.root)  # TODO group all seeds and sub-seeds
        last_preview_frame = call_render_preview(init, indexes.frame, last_preview_frame)
        _update_status_tracker(init.root, indexes.frame, init.args.anim_args)
        init.animation_mode.cleanup()


def _setup_opts(init, schedule):
    data = init.args.opts.data
    if init.has_img2img_fix_steps():
        # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
        data["img2img_fix_steps"] = False
    put_if_present(data, "CLIP_stop_at_last_layers", schedule.clipskip)
    put_if_present(data, "initial_noise_multiplier", schedule.noise_multiplier)
    put_if_present(data, "eta_ddim", schedule.eta_ddim)
    put_if_present(data, "eta_ancestral", schedule.eta_ancestral)


WEB_UI_SLEEP_DELAY = 0.1  # TODO make a WebUi class?


def _init_web_ui_job(init):
    state.job_count = init.args.anim_args.max_frames


def _update_web_ui_job(init, indexes):
    frame = indexes.frame + 1
    max_frames = init.args.anim_args.max_frames
    state.job = f"frame {frame}/{max_frames}"
    state.job_no = frame + 1
    if state.skipped:
        print("\n** PAUSED **")
        state.skipped = False
        while not state.skipped:
            time.sleep(WEB_UI_SLEEP_DELAY)
        print("** RESUMING **")


def _update_status_tracker(root, frame_idx, anim_args):
    JobStatusTracker().update_phase(root.job_id, phase="GENERATING", progress=frame_idx / anim_args.max_frames)
