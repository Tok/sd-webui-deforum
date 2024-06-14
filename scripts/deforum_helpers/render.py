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

import cv2
import numpy as np
from PIL import Image
# noinspection PyUnresolvedReferences
from modules.shared import opts, state

from .colors import maintain_colors
from .rendering.data import Turbo, Images, Indexes, Mask
from .rendering.data.initialization import RenderInit
from .rendering.data.step import Step, TweenStep
from .rendering.img_2_img_tubes import (
    frame_transformation_tube, conditional_color_match_tube , conditional_hybrid_video_after_generation_tube,
    contrast_transformation_tube, noise_transformation_tube, optical_flow_redo_tube)
from .rendering.util import generate_random_seed, memory_utils, filename_utils, web_ui_utils
from .rendering.util.call.gen import call_generate
from .rendering.util.call.video_and_audio import call_render_preview
from .rendering.util.image_utils import add_overlay_mask_if_active, force_to_grayscale_if_required
from .rendering.util.log_utils import print_animation_frame_info, print_optical_flow_info, print_redo_generation_info
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

        if turbo.is_emit_in_between_frames():
            emit_in_between_frames(init, indexes, step, turbo, images)

        images.color_match = Step.create_color_match_for_video(init, indexes)

        # transform image
        if images.has_previous():  # skipping 1st iteration
            frame_tube, contrast_tube, noise_tube = image_transformation_tubes(init, indexes, step, images, mask)
            images.previous, noised_image = run_tubes(frame_tube, contrast_tube, noise_tube, images.previous)
            init.update_sample_and_args_for_current_progression_step(step, noised_image)

        init.prepare_generation(indexes, step, mask)
        memory_utils.handle_vram_if_depth_is_predicted(init)

        # optical flow redo before generation
        optical_flow_redo_generation = init.optical_flow_redo_generation_if_not_in_preview_mode()
        is_redo_optical_flow = step.is_optical_flow_redo_before_generation(optical_flow_redo_generation, images)
        if is_redo_optical_flow:
            init.args.root.init_sample = do_optical_flow_redo_before_generation(init, indexes, step, images)
            gc.collect()

        # diffusion redo
        is_diffusion_redo = init.has_positive_diffusion_redo and images.has_previous() and step.init.has_strength()
        is_not_preview = init.is_not_in_motion_preview_mode()
        if is_diffusion_redo and is_not_preview:
            do_diffusion_redo(init, indexes, step, images)
            gc.collect()

        # generation
        image = call_generate(init, indexes.frame.i, step.schedule)

        if image is None:  # TODO throw error or log warning or something
            break

        # do hybrid video after generation
        image = conditional_hybrid_video_after_generation_tube(init, indexes, step)(image)

        # color matching on first frame is after generation, color match was collected earlier,
        # so we do an extra generation to avoid the corruption introduced by the color match of first output
        image = conditional_color_match_tube(init, indexes, images)(image)

        image = force_to_grayscale_if_required(init, image)  # TODO move to tubes?
        image = add_overlay_mask_if_active(init, image)

        # on strength 0, set color match to generation
        if init.is_do_color_match_conversion(step):
            images.color_match = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not init.animation_mode.has_video_input:
            images.previous = opencv_image

        next_frame, step.depth = progress_step(init, indexes, turbo, opencv_image, image, step.depth)
        indexes.update_frame(next_frame)

        state.assign_current_image(image)
        # may reassign init.args.args and/or root.seed_internal
        init.args.args.seed = next_seed(init.args.args, init.args.root)  # TODO group all seeds and sub-seeds
        last_preview_frame = call_render_preview(init, indexes.frame.i, last_preview_frame)
        web_ui_utils.update_status_tracker(init, indexes)
        init.animation_mode.unload_raft_and_depth_model()


def emit_in_between_frames(init, indexes, step, turbo, images):
    tween_frame_start_i = max(indexes.frame.start, indexes.frame.i - turbo.steps)
    emit_frames_between_index_pair(init, indexes, step, turbo, images, tween_frame_start_i, indexes.frame.i)


def emit_frames_between_index_pair(init, indexes, step, turbo, images, tween_frame_start_i, frame_i):
    """Emits tween frames (also known as turbo- or cadence-frames) between the provided indicis."""
    # TODO refactor until this works with just 2 args: RenderInit and a collection of immutable TweenStep objects.
    indexes.update_tween_start(turbo)

    tween_range = range(tween_frame_start_i, frame_i)

    # TODO Instead of indexes, pass a set of TweenStep objects to be processed instead of creating them in the loop.
    for tween_index in tween_range:
        # TODO tween index shouldn't really be updated and passed around like this here.
        #  ideally provide index data within an immutable TweenStep instance.
        indexes.update_tween(tween_index)  # TODO Nope

        tween_step = TweenStep.create(indexes)

        TweenStep.handle_synchronous_status_concerns(init, indexes, step, tween_step)
        TweenStep.process(init, indexes, step, turbo, images, tween_step)
        new_image = TweenStep.generate_and_save_frame(init, indexes, step, turbo, tween_step)

        images.previous = new_image  # TODO shouldn't


def image_transformation_tubes(init, indexes, step, images, mask):
    return (frame_transformation_tube(init, indexes, step, images),
            contrast_transformation_tube(init, step, mask),
            noise_transformation_tube(init, step))


def run_tubes(frame_tube, contrast_tube, noise_tube, original_image):
    transformed_image = frame_tube(original_image)
    contrasted_image = contrast_tube(transformed_image)
    noised_image = noise_tube(contrasted_image)
    return transformed_image, noised_image  # TODO separate


def generate_depth_maps_if_active(init):
    # TODO move all depth related stuff to new class.
    if init.args.anim_args.save_depth_maps:
        memory_utils.handle_vram_before_depth_map_generation(init)
        depth = init.depth_model.predict(opencv_image, init.args.anim_args.midas_weight, init.args.root.half_precision)
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
        save_image(image, 'PIL', filename, init.args.args, init.args.video_args, init.args.root)
        depth = generate_depth_maps_if_active(init)
        return idx.frame.i + 1, depth  # normal (i.e. 'non-turbo') step always increments by 1.


def do_optical_flow_redo_before_generation(init, indexes, step, images):
    stored_seed = init.args.args.seed  # keep original to reset it after executing the optical flow
    init.args.args.seed = generate_random_seed()  # set a new random seed
    print_optical_flow_info(init, optical_flow_redo_generation)  # TODO output temp seed?

    sample_image = call_generate(init, indexes.frame.i, step.schedule)
    optical_tube = optical_flow_redo_tube(init, optical_flow_redo_generation, images)
    transformed_sample_image = optical_tube(sample_image)

    init.args.args.seed = stored_seed  # restore stored seed
    return Image.fromarray(transformed_sample_image)


def do_diffusion_redo(init, indexes, step, images):
    stored_seed = init.args.args.seed
    last_diffusion_redo_index = int(init.args.anim_args.diffusion_redo)
    for n in range(0, last_diffusion_redo_index):
        print_redo_generation_info(init, n)
        init.args.args.seed = generate_random_seed()
        diffusion_redo_image = call_generate(init, indexes.frame.i, step.schedule)
        diffusion_redo_image = cv2.cvtColor(np.array(diffusion_redo_image), cv2.COLOR_RGB2BGR)
        # color match on last one only
        is_last_iteration = n == last_diffusion_redo_index
        if is_last_iteration:
            mode = init.args.anim_args.color_coherence
            diffusion_redo_image = maintain_colors(images.previous, images.color_match, mode)
        init.args.args.seed = stored_seed
        init.args.root.init_sample = Image.fromarray(cv2.cvtColor(diffusion_redo_image, cv2.COLOR_BGR2RGB))
