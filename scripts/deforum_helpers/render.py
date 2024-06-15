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
from .rendering.img_2_img_tubes import (conditional_color_match_tube, conditional_frame_transformation_tube,
                                        contrasted_noise_transformation_tube, optical_flow_redo_tube,
                                        frame_transformation_tube)
from .rendering.util import generate_random_seed, memory_utils, filename_utils, web_ui_utils
from .rendering.util.call.gen import call_generate
from .rendering.util.call.video_and_audio import call_render_preview
from .rendering.util.image_utils import save_and_return_frame
from .rendering.util.log_utils import (print_animation_frame_info, print_optical_flow_info,
                                       print_redo_generation_info, print_warning_generate_returned_no_image)
from .save_images import save_image
from .seed import next_seed


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    init = RenderInit.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    run_render_animation(init)


def run_render_animation(init):
    images = Images.create(init)
    turbo = Turbo.create(init)
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

        maybe_emit_in_between_frames(init, indexes, step, turbo, images)

        images.color_match = Step.create_color_match_for_video(init, indexes)
        images.previous = transform_and_update_noised_sample(init, indexes, step, images, mask)
        init.prepare_generation(init, indexes, step, mask)
        maybe_redo_optical_flow(init, indexes, step, images)
        maybe_redo_diffusion(init, indexes, step, images)

        image = call_generate(init, indexes.frame.i, step.schedule)
        if image is None:
            print_warning_generate_returned_no_image()
            break

        image = conditional_frame_transformation_tube(init, indexes, step, images)(image)
        images.color_match = conditional_color_match_tube(init, step)(image)
        next_frame, step.depth = progress_step(init, indexes, turbo, images, image, step.depth)
        indexes.update_frame(next_frame)
        state.assign_current_image(image)
        # may reassign init.args.args and/or root.seed_internal
        init.args.args.seed = next_seed(init.args.args, init.args.root)  # TODO group all seeds and sub-seeds
        last_preview_frame = call_render_preview(init, indexes.frame.i, last_preview_frame)
        web_ui_utils.update_status_tracker(init, indexes)
        init.animation_mode.unload_raft_and_depth_model()


def maybe_emit_in_between_frames(init, indexes, step, turbo, images):
    if turbo.is_emit_in_between_frames():
        tween_frame_start_i = max(indexes.frame.start, indexes.frame.i - turbo.steps)
        emit_frames_between_index_pair(init, indexes, step, turbo, images, tween_frame_start_i, indexes.frame.i)


def emit_frames_between_index_pair(init, indexes, last_step, turbo, images, tween_frame_start_i, frame_i):
    """Emits tween frames (also known as turbo- or cadence-frames) between the provided indices."""
    tween_range = range(tween_frame_start_i, frame_i)
    tween_indexes_list: List[Indexes] = TweenStep.create_indexes(indexes, tween_range)
    tween_steps: List[TweenStep] = TweenStep.create_steps(last_step, tween_indexes_list)
    indexes.update_tween_start(turbo)  # TODO...
    emit_tween_frames(init, tween_steps, turbo, images)


def emit_tween_frames(init, tween_steps, turbo, images):
    """Emits a tween frame for each provided tween_step."""
    for tween_step in tween_steps:
        tween_step.handle_synchronous_status_concerns(init)
        tween_step.process(init, turbo, images)  # side effects on turbo and on step

        new_image = tween_step.generate(init, turbo)
        # TODO pass depth instead of tween_step.indexes
        new_image = save_and_return_frame(init, tween_step.indexes, new_image)
        images.previous = new_image  # updating reference images to calculate hybrid motions in next iteration


def generate_depth_maps_if_active(init):
    # TODO move all depth related stuff to new class.
    if init.args.anim_args.save_depth_maps:
        memory_utils.handle_vram_before_depth_map_generation(init)
        depth = init.depth_model.predict(opencv_image, init.args.anim_args.midas_weight, init.args.root.half_precision)
        depth_filename = filename_utils.depth_frame(init, idx)
        init.depth_model.save(os.path.join(init.output_directory, depth_filename), depth)
        memory_utils.handle_vram_after_depth_map_generation(init)
        return depth


def progress_step(init, idx, turbo, images, image, depth):
    """Will progress frame or turbo-frame step and return next index and `depth`."""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if not init.animation_mode.has_video_input:
        images.previous = opencv_image
    if turbo.has_steps():
        return idx.frame.i + turbo.progress_step(idx, opencv_image), depth
    else:
        filename = filename_utils.frame(init, idx)
        save_image(image, 'PIL', filename, init.args.args, init.args.video_args, init.args.root)
        depth = generate_depth_maps_if_active(init)
        return idx.frame.i + 1, depth  # normal (i.e. 'non-turbo') step always increments by 1.


def transform_and_update_noised_sample(init, indexes, step, images, mask):
    if images.has_previous():  # skipping 1st iteration
        transformed_image = frame_transformation_tube(init, indexes, step, images)(images.previous)
        # TODO separate
        noised_image = contrasted_noise_transformation_tube(init, step, mask)(transformed_image)
        init.update_sample_and_args_for_current_progression_step(step, noised_image)
        return transformed_image
    else:
        return None


# Conditional Redoes
def maybe_redo_optical_flow(init, indexes, step, images):
    optical_flow_redo_generation = init.optical_flow_redo_generation_if_not_in_preview_mode()
    is_redo_optical_flow = step.is_optical_flow_redo_before_generation(optical_flow_redo_generation, images)
    if is_redo_optical_flow:
        init.args.root.init_sample = do_optical_flow_redo_before_generation(init, indexes, step, images)


def maybe_redo_diffusion(init, indexes, step, images):
    is_diffusion_redo = init.has_positive_diffusion_redo and images.has_previous() and step.init.has_strength()
    is_not_preview = init.is_not_in_motion_preview_mode()
    if is_diffusion_redo and is_not_preview:
        do_diffusion_redo(init, indexes, step, images)


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
