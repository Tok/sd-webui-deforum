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
from PIL import Image
# noinspection PyUnresolvedReferences
from modules.shared import opts, state

from .colors import maintain_colors
from .prompt import prepare_prompt
from .rendering.data import Turbo, Images, Indexes, Mask
from .rendering.data.initialization import RenderInit
from .rendering.data.step import Step, TweenStep
from .rendering.util import memory_utils, filename_utils, opt_utils, web_ui_utils
from .rendering.util.call.gen import call_generate
from .rendering.util.call.hybrid import call_get_flow_from_images, call_hybrid_composite
from .rendering.util.call.images import call_add_noise, call_get_mask_from_file_with_frame
from .rendering.util.call.mask import call_compose_mask_with_check, call_unsharp_mask
from .rendering.util.call.video_and_audio import call_render_preview, call_get_next_frame
from .rendering.util.image_utils import (
    add_overlay_mask_if_active, force_to_grayscale_if_required, save_cadence_frame)
from .rendering.util.log_utils import (
    print_animation_frame_info, print_tween_frame_info, print_init_frame_info, print_optical_flow_info,
    print_redo_generation_info)
from .rendering.util.utils import context
from .save_images import save_image
from .seed import next_seed


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    init = RenderInit.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    run_render_animation(init)


def run_render_animation_controlled(init):
    raise NotImplementedError("not implemented.")


def run_render_animation(init):
    images = Images.create(init)
    turbo = Turbo.create(init)  # state for interpolating between diffusion steps
    indexes = Indexes.create(init, turbo)
    mask = Mask.create(init, indexes.frame.i)  # reset mask vals as they are overwritten in the compose_mask algorithm
    web_ui_utils.init_job(init)
    last_preview_frame = 0
    while indexes.frame.i < init.args.anim_args.max_frames:
        memory_utils.handle_med_or_low_vram_before_step(init)
        print_animation_frame_info(init, indexes)
        web_ui_utils.update_job(init, indexes)
        step = Step.create(init, indexes)
        step.write_frame_subtitle(init, indexes, turbo)
        if turbo.is_emit_in_between_frames():  # TODO extract tween frame emition
            indexes.update_tween_start(turbo)
            for tween_index in range(indexes.tween.start, indexes.frame.i):

                indexes.update_tween(tween_index)
                web_ui_utils.update_progress_during_cadence(init, indexes)
                tween_step = TweenStep.create(indexes)

                turbo.do_optical_flow_cadence_setup_before_animation_warping(init, tween_step)

                step.write_frame_subtitle_if_active(init, indexes, opt_utils)
                print_tween_frame_info(init, indexes, tween_step.cadence_flow, tween_step.tween)

                step.update_depth_prediction(init, turbo)
                turbo.advance(init, indexes.tween.i, step.depth)

                turbo.do_hybrid_video_motion(init, indexes, images)

                img = tween_step.generate_tween_image(init, indexes, step, turbo)
                images.previous = img  # get images.previous during cadence

                # current image update for cadence frames (left commented because it doesn't currently update the preview)
                # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                save_cadence_frame(init, indexes, img)

                if init.args.anim_args.save_depth_maps:
                    dm_save_path = os.path.join(init.output_directory, filename_utils.tween_depth_frame(init, indexes))
                    init.depth_model.save(dm_save_path, step.depth)

        images.color_match = Step.create_color_match_for_video(init, indexes)
        # after 1st frame, images.previous exists
        if images.previous is not None:
            images.previous = step.apply_frame_warp_transform(init, indexes, images.previous)
            images.previous = step.do_hybrid_compositing_before_motion(init, indexes, images.previous)
            images.previous = Step.apply_hybrid_motion_ransac_transform(init, indexes, images, images.previous)
            images.previous = Step.apply_hybrid_motion_optical_flow(init, indexes, images, images.previous)
            images.previous = step.do_normal_hybrid_compositing_after_motion(init, indexes, images.previous)
            images.previous = Step.apply_color_matching(init, images, images.previous)
            images.previous = Step.transform_to_grayscale_if_active(init, images, images.previous)

            # apply scaling
            contrast_image = (images.previous * step.init.contrast).round().astype(np.uint8)
            # anti-blur
            if step.init.amount > 0:
                step.init.kernel_size()
                contrast_image = call_unsharp_mask(init, step, contrast_image, mask)
            # apply frame noising
            if init.args.args.use_mask or init.args.anim_args.use_noise_mask:
                init.root.noise_mask = call_compose_mask_with_check(
                    init, step.schedule.noise_mask_seq, mask.noise_vals, contrast_image)

            noised_image = call_add_noise(init, step, contrast_image)

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

        with context(init.animation_keys.deform_keys) as keys:
            if init.args.args.seed_behavior == 'schedule' or init.parseq_adapter.manages_seed():
                init.args.args.seed = int(keys.seed_schedule_series[indexes.frame.i])
            if init.args.anim_args.enable_checkpoint_scheduling:
                init.args.args.checkpoint = keys.checkpoint_schedule_series[indexes.frame.i]
            else:
                init.args.args.checkpoint = None

            # SubSeed scheduling
            if init.args.anim_args.enable_subseed_scheduling:
                init.root.subseed = int(keys.subseed_schedule_series[indexes.frame.i])
                init.root.subseed_strength = float(keys.subseed_strength_schedule_series[indexes.frame.i])
            if init.parseq_adapter.manages_seed():
                init.args.anim_args.enable_subseed_scheduling = True
                init.root.subseed = int(keys.subseed_schedule_series[indexes.frame.i])
                init.root.subseed_strength = keys.subseed_strength_schedule_series[indexes.frame.i]

        # set value back into the prompt - prepare and report prompt and seed
        init.args.args.prompt = prepare_prompt(init.args.args.prompt, init.args.anim_args.max_frames,
                                               init.args.args.seed, indexes.frame.i)
        # grab init image for current frame
        if init.animation_mode.has_video_input:
            init_frame = call_get_next_frame(init, indexes.frame.i, init.args.anim_args.video_init_path)
            print_init_frame_info(init_frame)
            init.args.args.init_image = init_frame
            init.args.args.init_image_box = None  # init_image_box not used in this case
            init.args.args.strength = max(0.0, min(1.0, step.init.strength))
        if init.args.anim_args.use_mask_video:
            mask_init_frame = call_get_next_frame(init, indexes.frame.i, init.args.anim_args.video_mask_path, True)
            temp_mask = call_get_mask_from_file_with_frame(init, mask_init_frame)
            init.args.args.mask_file = temp_mask
            init.root.noise_mask = temp_mask
            mask.vals['video_mask'] = temp_mask

        if init.args.args.use_mask:
            init.args.args.mask_image = call_compose_mask_with_check(
                init, step.schedule.mask_seq, mask.vals, init.root.init_sample)
            init.args.args.mask_image = compose_mask_with_check(init.root, init.args.args, step.schedule.mask_seq,
                                                                mask.vals, init.root.init_sample) \
                if init.root.init_sample is not None else None  # we need it only after the first frame anyway

        init.animation_keys.update(indexes.frame.i)
        opt_utils.setup(init, step.schedule)

        memory_utils.handle_vram_if_depth_is_predicted(init)
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
            _, temp_image_2 = call_hybrid_composite(init, indexes.frame.i, temp_image, step.init.hybrid_comp_schedules)
            image = Image.fromarray(cv2.cvtColor(temp_image_2, cv2.COLOR_BGR2RGB))

        # color matching on first frame is after generation, color match was collected earlier,
        # so we do an extra generation to avoid the corruption introduced by the color match of first output
        if indexes.frame.i == 0 and init.is_color_match_to_be_initialized(images.color_match):
            temp_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            temp_image = maintain_colors(temp_color, images.color_match, init.args.anim_args.color_coherence)
            image = Image.fromarray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))

        image = force_to_grayscale_if_required(init, image)
        image = add_overlay_mask_if_active(init, image)

        # on strength 0, set color match to generation
        if (((not init.args.anim_args.legacy_colormatch and not init.args.args.use_init)
             or (init.args.anim_args.legacy_colormatch and step.init.strength == 0))
                and init.args.anim_args.color_coherence not in ['Image', 'Video Input']):
            images.color_match = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not init.animation_mode.has_video_input:
            images.previous = opencv_image

        next_frame, step.depth = progress_step(init, indexes, turbo, opencv_image, image, step.depth)
        indexes.update_frame(next_frame)

        state.assign_current_image(image)
        # may reassign init.args.args and/or root.seed_internal
        init.args.args.seed = next_seed(init.args.args, init.root)  # TODO group all seeds and sub-seeds
        last_preview_frame = call_render_preview(init, indexes.frame.i, last_preview_frame)
        web_ui_utils.update_status_tracker(init, indexes)
        init.animation_mode.unload_raft_and_depth_model()


def emit_in_between_frames():
    raise NotImplemented("")


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
