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

import PIL
import cv2
import numpy as np
from PIL import Image, ImageOps
# noinspection PyUnresolvedReferences
from modules.shared import opts, state

from .colors import maintain_colors
from .composable_masks import compose_mask_with_check
from .hybrid_video import (
    image_transform_ransac, image_transform_optical_flow, abs_flow_to_rel_flow, rel_flow_to_abs_flow)
from .image_sharpening import unsharp_mask
from .masks import do_overlay_mask
from .noise import add_noise
from .prompt import prepare_prompt
from .rendering.data import Turbo, Images, Indexes, Mask
from .rendering.data.step import StepInit, Step, SubStep
from .rendering.data.initialization import RenderInit
from .rendering.util import memory_utils, filename_utils, opt_utils, web_ui_utils
from .rendering.util.call.anim import call_anim_frame_warp
from .rendering.util.call.gen import call_generate
from .rendering.util.call.hybrid import (
    call_get_flow_from_images, call_get_flow_for_hybrid_motion, call_get_flow_for_hybrid_motion_prev,
    call_get_matrix_for_hybrid_motion, call_get_matrix_for_hybrid_motion_prev, call_hybrid_composite)
from .rendering.util.call.images import call_get_mask_from_file_with_frame
from .rendering.util.call.subtitle import call_format_animation_params, call_write_frame_subtitle
from .rendering.util.call.video_and_audio import call_render_preview, call_get_next_frame
from .rendering.util.log_utils import (
    print_animation_frame_info, print_tween_frame_info, print_init_frame_info, print_optical_flow_info,
    print_redo_generation_info)
from .rendering.util.utils import context
from .save_images import save_image
from .seed import next_seed


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    init = RenderInit.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    # TODO method is temporarily torn apart to remove args from direct access in larger execution scope.
    run_render_animation(init)


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
    indexes = Indexes.create(init, turbo)
    mask = Mask.create(init, indexes.frame.i)  # reset mask vals as they are overwritten in the compose_mask algorithm

    web_ui_utils.init_job(init)
    last_preview_frame = 0
    while indexes.frame.i < init.args.anim_args.max_frames:
        web_ui_utils.update_job(init, indexes)
        print_animation_frame_info(init, indexes)

        # TODO Stuff to be moved into new Step class in `run_render_animation_controlled`:
        step = Step.create(init, indexes)

        memory_utils.handle_med_or_low_vram_before_step(init)

        if turbo.is_first_step_with_subtitles(init):
            params_to_print = opt_utils.generation_info_for_subtitles(init)
            params_string = call_format_animation_params(init, indexes.frame.i, params_to_print)
            call_write_frame_subtitle(init, indexes.frame.i, params_string)

        if turbo.is_emit_in_between_frames():
            indexes.tween.start = max(indexes.frame.start, indexes.frame.i - turbo.steps)
            cadence_flow = None
            # TODO stuff for new SubStep class to be used in `run_render_animation_controlled`
            for indexes.tween.i in range(indexes.tween.start, indexes.frame.i):
                web_ui_utils.update_progress_during_cadence(init, indexes)

                sub_step = SubStep.create(init, indexes)

                # optical flow cadence setup before animation warping
                if (init.args.anim_args.animation_mode in ['2D', '3D']
                        and init.args.anim_args.optical_flow_cadence != 'None'):
                    if init.animation_keys.deform_keys.strength_schedule_series[indexes.tween.start.i] > 0:
                        if cadence_flow is None and turbo.prev.image is not None and turbo.next.image is not None:
                            cadence_flow = call_get_flow_from_images(init, turbo.prev.image, turbo.next.image,
                                                                     init.args.anim_args.optical_flow_cadence) / 2
                            turbo.next.image = image_transform_optical_flow(turbo.next.image, -cadence_flow, 1)

                if opt_utils.is_generate_subtitles(init):
                    params_to_print = opt_utils.generation_info_for_subtitles(init)
                    params_string = call_format_animation_params(init, indexes.tween.i, params_to_print)
                    call_write_frame_subtitle(init, indexes.tween.i, params_string, sub_step.tween < 1.0)

                print_tween_frame_info(init, indexes, cadence_flow, sub_step.tween)

                if init.depth_model is not None:
                    assert (turbo.next.image is not None)
                    step.depth = init.depth_model.predict(turbo.next.image,
                                                     init.args.anim_args.midas_weight,
                                                     init.root.half_precision)

                turbo.advance(init, indexes.tween.i, step.depth)

                # hybrid video motion - warps turbo.prev.image or turbo.next.image to match motion
                if indexes.tween.i > 0:
                    if init.args.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                        if init.args.anim_args.hybrid_motion_use_prev_img:
                            matrix = call_get_matrix_for_hybrid_motion_prev(init, indexes.tween.i - 1, images.previous)
                            if turbo.is_advance_prev(indexes.tween.i):
                                turbo.prev.image = image_transform_ransac(turbo.prev.image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                            if turbo.is_advance_next(indexes.tween.i):
                                turbo.next.image = image_transform_ransac(turbo.next.image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                        else:
                            matrix = call_get_matrix_for_hybrid_motion(init, indexes.tween.i - 1)
                            if turbo.is_advance_prev(indexes.tween.i):
                                turbo.prev.image = image_transform_ransac(turbo.prev.image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                            if turbo.is_advance_next(indexes.tween.i):
                                turbo.next.image = image_transform_ransac(turbo.next.image, matrix,
                                                                          init.args.anim_args.hybrid_motion)
                    if init.args.anim_args.hybrid_motion in ['Optical Flow']:
                        if init.args.anim_args.hybrid_motion_use_prev_img:
                            flow = call_get_flow_for_hybrid_motion_prev(init, indexes.tween.i - 1, images.previous)
                            if turbo.is_advance_prev(indexes.tween.i):
                                turbo.prev.image = image_transform_optical_flow(turbo.prev.image, flow,
                                                                                step.init.flow_factor())
                            if turbo.is_advance_next(indexes.tween.i):
                                turbo.next.image = image_transform_optical_flow(turbo.next.image, flow,
                                                                                step.init.flow_factor())
                            init.animation_mode.prev_flow = flow
                        else:
                            flow = call_get_flow_for_hybrid_motion(init, indexes.tween.i - 1)
                            if turbo.is_advance_prev(indexes.tween.i):
                                turbo.prev.image = image_transform_optical_flow(turbo.prev.image, flow,
                                                                                step.init.flow_factor())
                            if turbo.is_advance_next(indexes.tween.i):
                                turbo.next.image = image_transform_optical_flow(turbo.next.image, flow,
                                                                                step.init.flow_factor())
                            init.animation_mode.prev_flow = flow

                # TODO cadence related transforms to be decoupled and handled in a 2nd pass
                # do optical flow cadence after animation warping
                with context(cadence_flow) as cf:
                    if cf is not None:
                        cf = abs_flow_to_rel_flow(cf, init.width(), init.height())
                        cf, _ = call_anim_frame_warp(init, indexes.tween.i, cf, step.depth)
                        cadence_flow_inc = rel_flow_to_abs_flow(cf, init.width(), init.height()) * sub_step.tween
                        if turbo.is_advance_prev(indexes.tween.i):
                            turbo.prev.image = image_transform_optical_flow(turbo.prev.image, cadence_flow_inc,
                                                                            step.init.cadence_flow_factor)
                        if turbo.is_advance_next(indexes.tween.i):
                            turbo.next.image = image_transform_optical_flow(turbo.next.image, cadence_flow_inc,
                                                                            step.init.cadence_flow_factor)

                turbo.prev.index = turbo.next.frame_idx = indexes.tween.i

                if turbo.prev.image is not None and sub_step.tween < 1.0:
                    img = turbo.prev.image * (1.0 - sub_step.tween) + turbo.next.image * sub_step.tween
                else:
                    img = turbo.next.image

                # intercept and override to grayscale
                if init.args.anim_args.color_force_grayscale:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # overlay mask
                if init.args.args.overlay_mask and (init.args.anim_args.use_mask_video or init.args.args.use_mask):
                    img = do_overlay_mask(init.args.args, init.args.anim_args, img, indexes.tween.i, True)

                # get images.previous during cadence
                images.previous = img

                # current image update for cadence frames (left commented because it doesn't currently update the preview)
                # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                # saving cadence frames
                filename = filename_utils.tween_frame(init, indexes)
                save_path = os.path.join(init.args.args.outdir, filename)
                cv2.imwrite(save_path, img)

                if init.args.anim_args.save_depth_maps:
                    dm_save_path = os.path.join(init.output_directory, filename_utils.tween_depth_frame(init, indexes))
                    init.depth_model.save(dm_save_path, step.depth)

        # get color match for video outside of images.previous conditional
        if init.args.anim_args.color_coherence == 'Video Input' and init.is_hybrid_available():
            if int(indexes.frame.i) % int(init.args.anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(preview_video_image_path(init, indexes))
                prev_vid_img = prev_vid_img.resize(init.dimensions(), PIL.Image.LANCZOS)
                images.color_match = np.asarray(prev_vid_img)
                images.color_match = cv2.cvtColor(images.color_match, cv2.COLOR_RGB2BGR)

        # after 1st frame, images.previous exists
        if images.previous is not None:
            # apply transforms to previous frame
            images.previous, step.depth = call_anim_frame_warp(init, indexes.frame.i, images.previous, None)

            # do hybrid compositing before motion
            if init.is_hybrid_composite_before_motion():
                # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
                _, images.previous = call_hybrid_composite(init, indexes.frame.i, images.previous,
                                                           step.init.hybrid_comp_schedules)

            # hybrid video motion - warps images.previous to match motion, usually to prepare for compositing
            with context(init.args.anim_args) as aa:
                if aa.hybrid_motion in ['Affine', 'Perspective']:
                    if aa.hybrid_motion_use_prev_img:
                        matrix = call_get_matrix_for_hybrid_motion_prev(init, indexes.frame.i - 1, images.previous)
                    else:
                        matrix = call_get_matrix_for_hybrid_motion(init, indexes.frame.i - 1)
                    images.previous = image_transform_ransac(images.previous, matrix, aa.hybrid_motion)
                if aa.hybrid_motion in ['Optical Flow']:
                    if aa.hybrid_motion_use_prev_img:
                        flow = call_get_flow_for_hybrid_motion_prev(init, indexes.frame.i - 1, images.previous)
                    else:
                        flow = call_get_flow_for_hybrid_motion(init, indexes.frame.i - 1)
                    images.previous = image_transform_optical_flow(images.previous, flow, step.init.flow_factor())
                    init.animation_mode.prev_flow = flow

            # do hybrid compositing after motion (normal)
            if init.is_normal_hybrid_composite():
                # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
                _, images.previous = call_hybrid_composite(init, indexes.frame.i, images.previous,
                                                           step.init.hybrid_comp_schedules)
            # apply color matching
            if init.has_color_coherence():
                if images.color_match is None:
                    images.color_match = images.previous.copy()
                else:
                    images.previous = maintain_colors(images.previous, images.color_match,
                                                      init.args.anim_args.color_coherence)

            # intercept and override to grayscale
            if init.args.anim_args.color_force_grayscale:
                images.previous = cv2.cvtColor(images.previous, cv2.COLOR_BGR2GRAY)
                images.previous = cv2.cvtColor(images.previous, cv2.COLOR_GRAY2BGR)

            # apply scaling
            contrast_image = (images.previous * step.init.contrast).round().astype(np.uint8)
            # anti-blur
            if step.init.amount > 0:
                step.init.kernel_size()
                contrast_image = unsharp_mask(contrast_image,
                                              (step.init.kernel, step.init.kernel),
                                              step.init.sigma,
                                              step.init.amount,
                                              step.init.threshold,
                                              mask.image if init.args.args.use_mask else None)
            # apply frame noising
            if init.args.args.use_mask or init.args.anim_args.use_noise_mask:
                init.root.noise_mask = compose_mask_with_check(init.root,
                                                               init.args.args,
                                                               step.schedule.noise_mask_seq,
                                                               mask.noise_vals,
                                                               Image.fromarray(cv2.cvtColor(contrast_image,
                                                                                            cv2.COLOR_BGR2RGB)))

            with context(init.args.anim_args) as aa:
                noised_image = add_noise(contrast_image, step.init.noise, init.args.args.seed, aa.noise_type,
                                         (aa.perlin_w, aa.perlin_h, aa.perlin_octaves, aa.perlin_persistence),
                                         init.root.noise_mask, init.args.args.invert_mask)

            # use transformed previous frame as init for current
            init.args.args.use_init = True
            init.root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
            init.args.args.strength = max(0.0, min(1.0, step.init.strength))

        init.args.args.scale = step.init.scale

        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        init.args.args.pix2pix_img_cfg_scale = float(
            init.animation_keys.deform_keys.pix2pix_img_cfg_scale_series[indexes.frame.i])

        # grab prompt for current frame
        init.args.args.prompt = init.prompt_series[indexes.frame.i]

        with context(init.args) as ia:
            with context(init.animation_keys.deform_keys) as keys:
                # FIXME? check ia.args.seed_behavior
                if ia.args.seed_behavior == 'schedule' or init.parseq_adapter.manages_seed():
                    ia.args.seed = int(keys.seed_schedule_series[indexes.frame.i])  # TODO recontextualize frame index
                if ia.anim_args.enable_checkpoint_scheduling:
                    ia.args.checkpoint = keys.checkpoint_schedule_series[indexes.frame.i]
                else:
                    ia.args.checkpoint = None

                # SubSeed scheduling
                if ia.anim_args.enable_subseed_scheduling:
                    init.root.subseed = int(keys.subseed_schedule_series[indexes.frame.i])
                    init.root.subseed_strength = float(keys.subseed_strength_schedule_series[indexes.frame.i])
                if init.parseq_adapter.manages_seed():
                    init.args.anim_args.enable_subseed_scheduling = True
                    init.root.subseed = int(keys.subseed_schedule_series[indexes.frame.i])
                    init.root.subseed_strength = keys.subseed_strength_schedule_series[indexes.frame.i]

            # set value back into the prompt - prepare and report prompt and seed
            ia.args.prompt = prepare_prompt(ia.args.prompt, ia.anim_args.max_frames, ia.args.seed, indexes.frame.i)
            # grab init image for current frame
            if init.animation_mode.has_video_input:
                init_frame = call_get_next_frame(init, indexes.frame.i, ia.anim_args.video_init_path)
                print_init_frame_info(init_frame)
                ia.args.init_image = init_frame
                ia.args.init_image_box = None  # init_image_box not used in this case
                ia.args.strength = max(0.0, min(1.0, step.init.strength))
            if ia.anim_args.use_mask_video:
                mask_init_frame = call_get_next_frame(init, indexes.frame.i, ia.anim_args.video_mask_path, True)
                temp_mask = call_get_mask_from_file_with_frame(init, mask_init_frame)
                ia.args.mask_file = temp_mask
                init.root.noise_mask = temp_mask
                mask.vals['video_mask'] = temp_mask

            if ia.args.use_mask:
                # TODO figure why this is different from mask.image
                ia.args.mask_image = compose_mask_with_check(init.root, ia.args, step.schedule.mask_seq,
                                                             mask.vals, init.root.init_sample) \
                    if init.root.init_sample is not None else None  # we need it only after the first frame anyway

        init.animation_keys.update(indexes.frame.i)
        opt_utils.setup(init, step.schedule)

        memory_utils.handle_vram_if_depth_is_predicted(init)

        # TODO try init early, also see "call_get_flow_from_images"
        optical_flow_redo_generation = init.args.anim_args.optical_flow_redo_generation \
            if not init.args.args.motion_preview_mode else 'None'

        # optical flow redo before generation
        if optical_flow_redo_generation != 'None' and images.previous is not None and step.init.strength > 0:
            stored_seed = init.args.args.seed
            init.args.args.seed = random.randint(0, 2 ** 32 - 1)  # TODO move elsewhere
            print_optical_flow_info(init, optical_flow_redo_generation)
            with context(call_generate(init, indexes.frame.i, step.schedule)) as img:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                disposable_flow = call_get_flow_from_images(init, images.previous, img, optical_flow_redo_generation)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = image_transform_optical_flow(img, disposable_flow, step.init.redo_flow_factor)
                init.args.args.seed = stored_seed  # TODO check if (or make) unnecessary and group seeds
                init.root.init_sample = Image.fromarray(img)
                disposable_image = img  # TODO refactor
                del (img, disposable_flow, stored_seed)
                gc.collect()

        # diffusion redo
        if (int(init.args.anim_args.diffusion_redo) > 0
                and images.previous is not None and step.init.strength > 0
                and not init.args.args.motion_preview_mode):
            stored_seed = init.args.args.seed
            for n in range(0, int(init.args.anim_args.diffusion_redo)):
                print_redo_generation_info(init, n)
                init.args.args.seed = random.randint(0, 2 ** 32 - 1)  # TODO move elsewhere
                disposable_image = call_generate(init, indexes.frame.i, step.schedule)
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
        image = call_generate(init, indexes.frame.i, step.schedule)

        if image is None:
            break

        # do hybrid video after generation
        if indexes.frame.i > 0 and init.is_hybrid_composite_after_generation():
            temp_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # TODO test, returned args seem unchanged, so might as well be ignored here (renamed to _)
            _, temp_image_2 = call_hybrid_composite(init, indexes.frame.i, temp_image, step.init.hybrid_comp_schedules)
            image = Image.fromarray(cv2.cvtColor(temp_image_2, cv2.COLOR_BGR2RGB))

        # color matching on first frame is after generation, color match was collected earlier,
        # so we do an extra generation to avoid the corruption introduced by the color match of first output
        if indexes.frame.i == 0 and init.is_color_match_to_be_initialized(images.color_match):
            temp_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            temp_image = maintain_colors(temp_color, images.color_match, init.args.anim_args.color_coherence)
            image = Image.fromarray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))

        # intercept and override to grayscale
        if init.args.anim_args.color_force_grayscale:
            image = ImageOps.grayscale(image)
            image = ImageOps.colorize(image, black="black", white="white")

        # overlay mask
        if init.args.args.overlay_mask and (init.args.anim_args.use_mask_video or init.is_use_mask):
            image = do_overlay_mask(init.args.args, init.args.anim_args, image, indexes.frame.i)

        # on strength 0, set color match to generation
        if (((not init.args.anim_args.legacy_colormatch and not init.args.args.use_init)
             or (init.args.anim_args.legacy_colormatch and step.init.strength == 0))
                and init.args.anim_args.color_coherence not in ['Image', 'Video Input']):
            images.color_match = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not init.animation_mode.has_video_input:
            images.previous = opencv_image

        indexes.frame.i, step.depth = progress_step(init, indexes, turbo, opencv_image, image, step.depth)
        state.assign_current_image(image)
        # may reassign init.args.args and/or root.seed_internal
        init.args.args.seed = next_seed(init.args.args, init.root)  # TODO group all seeds and sub-seeds
        last_preview_frame = call_render_preview(init, indexes.frame.i, last_preview_frame)
        web_ui_utils.update_status_tracker(init, indexes)
        init.animation_mode.unload_raft_and_depth_model()


def generate_depth_maps_if_active(init):
    # TODO move all depth related stuff to new class.
    if init.args.anim_args.save_depth_maps:
        memory_utils.handle_vram_before_depth_map_generation(init)
        depth = init.depth_model.predict(opencv_image, init.args.anim_args.midas_weight, init.root.half_precision)
        depth_filename = filename_utils.depth_frame(init, idx)
        init.depth_model.save(os.path.join(init.output_directory, depth_filename), depth)
        memory_utils.handle_vram_after_depth_map_generation(init)
        return depth


def progress_step(init, idx, turbo, opencv_image, image, depth):
    """Will progress frame or turbo-frame step and return next index and `depth`."""
    if turbo.has_steps():
        return idx.frame.i + turbo.progress_step(idx, opencv_image), depth
    else:
        filename = filename_utils.frame(init, idx)
        save_image(image, 'PIL', filename, init.args.args, init.args.video_args, init.root)
        depth = generate_depth_maps_if_active(init)
        return idx.frame.i + 1, depth  # normal (i.e. 'non-turbo') step always increments by 1.
