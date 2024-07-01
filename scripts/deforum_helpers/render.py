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
from pathlib import Path

import PIL
# noinspection PyUnresolvedReferences
from modules.shared import cmd_opts, opts, progress_print_out, state
from tqdm import tqdm

from .rendering import img_2_img_tubes
from .rendering.data.render_data import RenderData
from .rendering.data.step import KeyIndexDistribution, KeyStep
from .rendering.util import filename_utils, log_utils, memory_utils, web_ui_utils


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    render_data = RenderData.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    run_render_animation(render_data)


@log_utils.with_suppressed_table_printing
def run_render_animation(data: RenderData):
    web_ui_utils.init_job(data)
    _ = data.turbo.find_start(data)  # called because it sets up step vars
    start_index = 0
    max_frames = data.args.anim_args.max_frames

    key_steps = KeyStep.create_all_steps(data, start_index, KeyIndexDistribution.PARSEQ_ONLY)
    for key_step in key_steps:
        # Check if resume
        filename = filename_utils.frame_filename(data, key_step.i)
        full_path = Path(data.output_directory) / filename
        is_file_existing = os.path.exists(full_path)
        if is_file_existing:
            log_utils.warn(f"Frame {filename} exists, skipping to next key frame.")
            key_step.render_data.args.args.seed = key_step.next_seed()
            continue

        memory_utils.handle_med_or_low_vram_before_step(data)
        web_ui_utils.update_job(data)

        is_step_with_tweens = len(key_step.tweens) > 0
        if is_step_with_tweens:  # emit tweens
            # setup variable pseudo cadence
            data.parseq_adapter.cadence = len(key_step.tweens)
            data.parseq_adapter.a1111_cadence = len(key_step.tweens)
            data.args.anim_args.diffusion_cadence = len(key_step.tweens)
            data.args.anim_args.optical_flow_cadence = len(key_step.tweens)
            data.args.anim_args.cadence_flow_factor_schedule = len(key_step.tweens)
            # data.parseq_adapter.print_parseq_table()

            log_utils.print_tween_frame_from_to_info(key_step)
            grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
            overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
            tq = tqdm(key_step.tweens, position=1, desc="Tweens progress", file=progress_print_out,
                      disable=cmd_opts.disable_console_progressbars, leave=False, colour='#FFA468')
            # FIXME disable progress bar is verbose console output is activated:
            # tq = key_step.tweens
            [tween.emit_frame(key_step, grayscale_tube, overlay_mask_tube) for tween in tq]

        log_utils.print_animation_frame_info(key_step.i, max_frames)
        key_step.maybe_write_frame_subtitle()

        frame_tube = img_2_img_tubes.frame_transformation_tube
        contrasted_noise_tube = img_2_img_tubes.contrasted_noise_transformation_tube
        key_step.prepare_generation(frame_tube, contrasted_noise_tube)

        image = key_step.do_generation()
        if image is None:
            log_utils.print_warning_generate_returned_no_image()
            break

        # TODO move check
        if type(image) is not PIL.Image.Image:  # check is required when resuming from timestring
            image = img_2_img_tubes.conditional_frame_transformation_tube(key_step)(image)
        key_step.render_data.images.color_match = img_2_img_tubes.conditional_color_match_tube(key_step)(image)

        key_step.progress_and_save(image)
        state.assign_current_image(image)

        key_step.render_data.args.args.seed = key_step.next_seed()

        key_step.update_render_preview()
        web_ui_utils.update_status_tracker(key_step.render_data)

    data.animation_mode.unload_raft_and_depth_model()
