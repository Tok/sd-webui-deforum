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

# noinspection PyUnresolvedReferences
from modules.shared import opts, state

from .rendering import img_2_img_tubes
from .rendering.data.render_data import RenderData
from .rendering.data.step import KeyStep, Tween
from .rendering.util import memory_utils, web_ui_utils
from .rendering.util.log_utils import print_animation_frame_info, print_warning_generate_returned_no_image


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    render_data = RenderData.create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
    run_render_animation(render_data)


def run_render_animation(data: RenderData):
    web_ui_utils.init_job(data)
    key_steps = KeyStep.create_all_steps(data)  # TODO precalculate and add `tweens: List[Tween]` to KeyStep
    for key_step in key_steps:
        memory_utils.handle_med_or_low_vram_before_step(data)
        print_animation_frame_info(key_step.render_data)
        web_ui_utils.update_job(data)

        # TODO emit precalculated `key_step.tweens` if there are any
        grayscale_tube = img_2_img_tubes.conditional_force_tween_to_grayscale_tube
        overlay_mask_tube = img_2_img_tubes.conditional_add_overlay_mask_tube
        _ = Tween.maybe_emit_in_between_frames(key_step, grayscale_tube, overlay_mask_tube)

        key_step.maybe_write_frame_subtitle()

        frame_tube = img_2_img_tubes.frame_transformation_tube
        contrasted_noise_tube = img_2_img_tubes.contrasted_noise_transformation_tube
        key_step.prepare_generation(frame_tube, contrasted_noise_tube)
        image = key_step.do_generation()
        if image is None:
            print_warning_generate_returned_no_image()
            break

        image = img_2_img_tubes.conditional_frame_transformation_tube(key_step)(image)
        key_step.render_data.images.color_match = img_2_img_tubes.conditional_color_match_tube(key_step)(image)

        key_step.progress_and_save(image)
        state.assign_current_image(image)
        key_step.render_data.args.args.seed = key_step.next_seed()

        key_step.update_render_preview()
        web_ui_utils.update_status_tracker(key_step.render_data)
        key_step.render_data.animation_mode.unload_raft_and_depth_model()
