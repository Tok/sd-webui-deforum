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
from .rendering.data import Indexes
from .rendering.data.render_data import RenderData
from .rendering.data.step import Step, TweenStep
from .rendering.img_2_img_tubes import (conditional_color_match_tube, conditional_frame_transformation_tube,
                                        contrasted_noise_transformation_tube, optical_flow_redo_tube,
                                        frame_transformation_tube)
from .rendering.util import generate_random_seed, memory_utils, filename_utils, web_ui_utils
from .rendering.util.call.gen import call_generate
from .rendering.util.call.video_and_audio import call_render_preview
from .rendering.util.image_utils import save_and_return_frame
from .rendering.util.log_utils import (print_optical_flow_info, print_redo_generation_info,
                                       print_warning_generate_returned_no_image)
from .save_images import save_image
from .seed import next_seed


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    render_data = RenderData.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    run_render_animation(render_data)


def run_render_animation(data: RenderData):
    web_ui_utils.init_job(data)
    last_preview_frame = 0
    while data.indexes.frame.i < data.args.anim_args.max_frames:
        step = Step.do_start_and_create(data)  # TODO step should have an immutable init?
        step.write_frame_subtitle(data)  # TODO move step concerns from init to step..

        maybe_emit_in_between_frames(data, step)

        data.images.color_match = Step.create_color_match_for_video(data)  # TODO move to step?
        data.images.previous = transform_and_update_noised_sample(data, step)  # TODO move to step?
        data.prepare_generation(data, step)  # TODO move to step?
        maybe_redo_optical_flow(data, step)  # TODO move to step?
        maybe_redo_diffusion(data, step)  # TODO move to step?

        image = call_generate(data, step)  # TODO move to step?
        if image is None:
            print_warning_generate_returned_no_image()
            break

        image = conditional_frame_transformation_tube(data, step)(image)  # TODO move to step?
        data.images.color_match = conditional_color_match_tube(data, step)(image)  # TODO move to step?
        next_frame, step.depth = progress_step(data, image, step.depth)  # TODO move to step?
        data.indexes.update_frame(next_frame)
        state.assign_current_image(image)
        # may reassign init.args.args and/or root.seed_internal
        data.args.args.seed = next_seed(data.args.args, data.args.root)  # TODO group all seeds and sub-seeds
        last_preview_frame = call_render_preview(data, last_preview_frame)
        web_ui_utils.update_status_tracker(data)
        data.animation_mode.unload_raft_and_depth_model()


def maybe_emit_in_between_frames(init, step):
    if init.turbo.is_emit_in_between_frames():
        tween_frame_start_i = max(init.indexes.frame.start, init.indexes.frame.i - init.turbo.steps)
        emit_frames_between_index_pair(init, step, tween_frame_start_i, init.indexes.frame.i)


def emit_frames_between_index_pair(init, last_step, tween_frame_start_i, frame_i):
    """Emits tween frames (also known as turbo- or cadence-frames) between the provided indices."""
    tween_range = range(tween_frame_start_i, frame_i)
    tween_indexes_list: List[Indexes] = TweenStep.create_indexes(init.indexes, tween_range)
    tween_steps: List[TweenStep] = TweenStep.create_steps(last_step, tween_indexes_list)
    init.indexes.update_tween_start(init.turbo)  # TODO...
    emit_tween_frames(init, tween_steps)


def emit_tween_frames(init, tween_steps):
    """Emits a tween frame for each provided tween_step."""
    for tween_step in tween_steps:
        tween_step.handle_synchronous_status_concerns(init)
        tween_step.process(init)  # side effects on turbo and on step

        new_image = tween_step.generate(init)
        # TODO pass depth instead of tween_step.indexes
        new_image = save_and_return_frame(init, tween_step.indexes, new_image)
        init.images.previous = new_image  # updating reference images to calculate hybrid motions in next iteration


def generate_depth_maps_if_active(init):
    # TODO move all depth related stuff to new class.
    if init.args.anim_args.save_depth_maps:
        memory_utils.handle_vram_before_depth_map_generation(init)
        depth = init.depth_model.predict(opencv_image, init.args.anim_args.midas_weight, init.args.root.half_precision)
        depth_filename = filename_utils.depth_frame(init, idx)
        init.depth_model.save(os.path.join(init.output_directory, depth_filename), depth)
        memory_utils.handle_vram_after_depth_map_generation(init)
        return depth


def progress_step(init, image, depth):
    """Will progress frame or turbo-frame step and return next index and `depth`."""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if not init.animation_mode.has_video_input:
        init.images.previous = opencv_image
    if init.turbo.has_steps():
        return init.indexes.frame.i + init.turbo.progress_step(init.indexes, opencv_image), depth
    else:
        filename = filename_utils.frame(init, init.indexes)
        save_image(image, 'PIL', filename, init.args.args, init.args.video_args, init.args.root)
        depth = generate_depth_maps_if_active(init)
        return init.indexes.frame.i + 1, depth  # normal (i.e. 'non-turbo') step always increments by 1.


def transform_and_update_noised_sample(init, step):
    if init.images.has_previous():  # skipping 1st iteration
        transformed_image = frame_transformation_tube(init, step)(init.images.previous)
        # TODO separate
        noised_image = contrasted_noise_transformation_tube(init, step)(transformed_image)
        init.update_sample_and_args_for_current_progression_step(step, noised_image)
        return transformed_image
    else:
        return None


# Conditional Redoes
def maybe_redo_optical_flow(init, step):
    optical_flow_redo_generation = init.optical_flow_redo_generation_if_not_in_preview_mode()
    is_redo_optical_flow = step.is_optical_flow_redo_before_generation(optical_flow_redo_generation, init.images)
    if is_redo_optical_flow:
        init.args.root.init_sample = do_optical_flow_redo_before_generation(init, step)


def maybe_redo_diffusion(init, step):
    is_diffusion_redo = init.has_positive_diffusion_redo and init.images.has_previous() and step.step_data.has_strength()
    is_not_preview = init.is_not_in_motion_preview_mode()
    if is_diffusion_redo and is_not_preview:
        do_diffusion_redo(init, step)


def do_optical_flow_redo_before_generation(init, step):
    stored_seed = init.args.args.seed  # keep original to reset it after executing the optical flow
    init.args.args.seed = generate_random_seed()  # set a new random seed
    print_optical_flow_info(init, optical_flow_redo_generation)  # TODO output temp seed?

    sample_image = call_generate(init, init.indexes.frame.i, step.schedule)
    optical_tube = optical_flow_redo_tube(init, optical_flow_redo_generation)
    transformed_sample_image = optical_tube(sample_image)

    init.args.args.seed = stored_seed  # restore stored seed
    return Image.fromarray(transformed_sample_image)


def do_diffusion_redo(init, step):
    stored_seed = init.args.args.seed
    last_diffusion_redo_index = int(init.args.anim_args.diffusion_redo)
    for n in range(0, last_diffusion_redo_index):
        print_redo_generation_info(init, n)
        init.args.args.seed = generate_random_seed()
        diffusion_redo_image = call_generate(init, step)
        diffusion_redo_image = cv2.cvtColor(np.array(diffusion_redo_image), cv2.COLOR_RGB2BGR)
        # color match on last one only
        is_last_iteration = n == last_diffusion_redo_index
        if is_last_iteration:
            mode = init.args.anim_args.color_coherence
            diffusion_redo_image = maintain_colors(init.images.previous, init.images.color_match, mode)
        init.args.args.seed = stored_seed
        init.args.root.init_sample = Image.fromarray(cv2.cvtColor(diffusion_redo_image, cv2.COLOR_BGR2RGB))
