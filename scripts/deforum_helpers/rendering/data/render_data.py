import os
from dataclasses import dataclass
from typing import Any

import cv2
import numexpr
import numpy as np
import pandas as pd
from PIL import Image

from .anim import AnimationKeys, AnimationMode
from .images import Images
from .indexes import Indexes
from .mask import Mask
from .subtitle import Srt
from .turbo import Turbo
from ..util import memory_utils, opt_utils
from ..util.call.mask import call_compose_mask_with_check
from ..util.call.video_and_audio import call_get_next_frame
from ...args import DeforumArgs, DeforumAnimArgs, LoopArgs, ParseqArgs, RootArgs
from ...deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from ...depth import DepthModel
from ...generate import (isJson)
from ...parseq_adapter import ParseqAdapter
from ...prompt import prepare_prompt
from ...settings import save_settings_from_animation_run


@dataclass(init=True, frozen=True, repr=False, eq=False)
class RenderInitArgs:
    args: DeforumArgs = None
    parseq_args: ParseqArgs = None
    anim_args: DeforumAnimArgs = None
    video_args: Any = None
    controlnet_args: Any = None
    loop_args: LoopArgs = None
    opts: Any = None  # opts were passed from modules.shared.  # TODO import and access directly?
    root: RootArgs = None


@dataclass(init=True, frozen=True, repr=False, eq=False)
class RenderData:
    """The purpose of this class is to group and control all data used in render_animation"""
    images: Images | None  # TODO rename to reference_images?
    turbo: Turbo | None
    indexes: Indexes | None
    mask: Mask | None
    seed: int
    args: RenderInitArgs
    parseq_adapter: Any
    srt: Any
    animation_keys: AnimationKeys
    animation_mode: AnimationMode
    prompt_series: Any
    depth_model: Any
    output_directory: str
    is_use_mask: bool

    @staticmethod
    def create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root) -> 'RenderData':
        ri_args = RenderInitArgs(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)

        output_directory = args.outdir
        is_use_mask = args.use_mask
        parseq_adapter = RenderData.create_parseq_adapter(ri_args)
        srt = Srt.create_if_active(opts.data, output_directory, root.timestring, video_args.fps)
        animation_keys = AnimationKeys.from_args(ri_args, parseq_adapter, args.seed)
        animation_mode = AnimationMode.from_args(ri_args)
        prompt_series = RenderData.select_prompts(parseq_adapter, anim_args, animation_keys, root)
        depth_model = RenderData.create_depth_model_and_enable_depth_map_saving_if_active(
            animation_mode, root, anim_args, args)

        # Temporary instance only exists for using it to easily create other objects required by the actual instance.
        # Feels slightly awkward, but it's probably not worth optimizing since only 1st and gc can take care of it fine.
        incomplete_init = RenderData(None, None, None, None, args.seed, ri_args, parseq_adapter, srt, animation_keys,
                                     animation_mode, prompt_series, depth_model, output_directory, is_use_mask)
        images = Images.create(incomplete_init)
        turbo = Turbo.create(incomplete_init)
        indexes = Indexes.create(incomplete_init, turbo)
        mask = Mask.create(incomplete_init, indexes.frame.i)

        instance = RenderData(images, turbo, indexes, mask, args.seed, ri_args, parseq_adapter, srt, animation_keys,
                              animation_mode, prompt_series, depth_model, output_directory, is_use_mask)
        RenderData.init_looper_if_active(args, loop_args)
        RenderData.handle_controlnet_video_input_frames_generation(controlnet_args, args, anim_args)
        RenderData.create_output_directory_for_the_batch(args.outdir)
        RenderData.save_settings_txt(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)
        RenderData.maybe_resume_from_timestring(anim_args, root)
        return instance

    def is_3d(self):
        return self.args.anim_args.animation_mode == '3D'

    def is_3d_or_2d(self):
        return self.args.anim_args.animation_mode in ['2D', '3D']

    def has_optical_flow_cadence(self):
        return self.args.anim_args.optical_flow_cadence != 'None'

    def is_3d_with_med_or_low_vram(self):
        return self.is_3d() and memory_utils.is_low_or_med_vram()

    def width(self) -> int:
        return self.args.args.W

    def height(self) -> int:
        return self.args.args.H

    def dimensions(self) -> tuple[int, int]:
        return self.width(), self.height()

    # TODO group hybrid stuff elsewhere
    def is_hybrid_composite(self) -> bool:
        return self.args.anim_args.hybrid_composite != 'None'

    def is_normal_hybrid_composite(self) -> bool:
        return self.args.anim_args.hybrid_composite == 'Normal'

    def has_hybrid_motion(self) -> bool:
        return self.args.anim_args.hybrid_motion in ['Optical Flow', 'Affine', 'Perspective']

    def is_hybrid_available(self) -> bool:
        return self.is_hybrid_composite() or self.has_hybrid_motion()

    def is_hybrid_composite_before_motion(self) -> bool:
        return self.args.anim_args.hybrid_composite == 'Before Motion'

    def is_hybrid_composite_after_generation(self) -> bool:
        return self.args.anim_args.hybrid_composite == 'After Generation'

    def is_initialize_color_match(self, color_match_sample):
        """Determines whether to initialize color matching based on the given conditions."""
        has_video_input = self.args.anim_args.color_coherence == 'Video Input' and self.is_hybrid_available()
        has_image_color_coherence = self.args.anim_args.color_coherence == 'Image'
        has_any_color_sample = color_match_sample is not None  # TODO extract to own method?
        has_coherent_non_legacy_color_match = (self.args.anim_args.color_coherence != 'None'
                                               and not self.args.anim_args.legacy_colormatch)
        has_sample_and_match = has_any_color_sample and has_coherent_non_legacy_color_match
        return has_video_input or has_image_color_coherence or has_sample_and_match

    def has_color_coherence(self):
        return self.args.anim_args.color_coherence != 'None'

    def has_non_video_or_image_color_coherence(self):
        return self.args.anim_args.color_coherence not in ['Image', 'Video Input']

    def is_resuming_from_timestring(self):
        return self.args.anim_args.resume_from_timestring

    def has_video_input(self):
        return self.animation_mode.has_video_input

    def has_img2img_fix_steps(self):
        return 'img2img_fix_steps' in self.args.opts.data and self.args.opts.data["img2img_fix_steps"]

    def cadence(self) -> int:
        return int(self.args.anim_args.diffusion_cadence)

    def _has_init_image(self) -> bool:
        return self.args.args.init_image is not None and self.args.args.init_image != ''

    def _has_init_box(self) -> bool:
        return self.args.args.init_image_box is not None

    def _has_init_image_or_box(self) -> bool:
        return self._has_init_image() or self._has_init_box()

    def is_using_init_image_or_box(self) -> bool:
        return self.args.args.use_init and self._has_init_image_or_box()

    def is_not_in_motion_preview_mode(self):
        return not self.args.args.motion_preview_mode

    def color_coherence_mode(self):
        return self.args.anim_args.color_coherence

    def diffusion_redo(self):
        return self.args.anim_args.diffusion_redo

    def diffusion_redo_as_int(self):
        return int(self.diffusion_redo())

    def has_positive_diffusion_redo(self):
        return self.diffusion_redo_as_int() > 0

    def optical_flow_redo_generation(self):
        return self.args.anim_args.optical_flow_redo_generation

    def optical_flow_redo_generation_if_not_in_preview_mode(self):
        is_not_preview = self.is_not_in_motion_preview_mode()
        return self.optical_flow_redo_generation() if is_not_preview else 'None'  # don't replace with None value

    def is_do_color_match_conversion(self, step):
        is_legacy_cm = self.args.anim_args.legacy_colormatch
        is_use_init = self.args.args.use_init
        is_not_legacy_with_use_init = not is_legacy_cm and not is_use_init
        is_legacy_cm_without_strength = is_legacy_cm and step.step_data.strength == 0
        is_maybe_special_legacy = is_not_legacy_with_use_init or is_legacy_cm_without_strength
        return is_maybe_special_legacy and self.has_non_video_or_image_color_coherence()

    def update_sample_and_args_for_current_progression_step(self, step, noised_image):
        # use transformed previous frame as init for current
        self.args.args.use_init = True
        self.args.root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
        self.args.args.strength = max(0.0, min(1.0, step.step_data.strength))

    def update_some_args_for_current_step(self, step, i):
        keys = self.animation_keys.deform_keys
        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        self.args.args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[i])
        self.args.args.prompt = self.prompt_series[i]  # grab prompt for current frame
        self.args.args.scale = step.step_data.scale

    def update_seed_and_checkpoint_for_current_step(self, i):
        keys = self.animation_keys.deform_keys
        is_seed_scheduled = self.args.args.seed_behavior == 'schedule'
        is_seed_managed = self.parseq_adapter.manages_seed()
        is_seed_scheduled_or_managed = is_seed_scheduled or is_seed_managed
        if is_seed_scheduled_or_managed:
            self.args.args.seed = int(keys.seed_schedule_series[i])
        self.args.args.checkpoint = keys.checkpoint_schedule_series[i] \
            if self.args.anim_args.enable_checkpoint_scheduling else None

    def update_sub_seed_schedule_for_current_step(self, i):
        keys = self.animation_keys.deform_keys
        is_subseed_scheduling_enabled = self.args.anim_args.enable_subseed_scheduling
        is_seed_managed_by_parseq = self.parseq_adapter.manages_seed()
        if is_subseed_scheduling_enabled or is_seed_managed_by_parseq:
            self.args.root.subseed = int(keys.subseed_schedule_series[i])
        if is_subseed_scheduling_enabled and not is_seed_managed_by_parseq:
            self.args.root.subseed_strength = float(keys.subseed_strength_schedule_series[i])
        if is_seed_managed_by_parseq:
            self.args.root.subseed_strength = keys.subseed_strength_schedule_series[i]
            self.args.anim_args.enable_subseed_scheduling = True  # TODO should be enforced in init, not here.

    def prompt_for_current_step(self, i):
        """returns value to be set back into the prompt"""
        prompt = self.args.args.prompt
        max_frames = self.args.anim_args.max_frames
        seed = self.args.args.seed
        return prepare_prompt(prompt, max_frames, seed, i)

    def _update_video_input_for_current_frame(self, i, step):
        video_init_path = self.args.anim_args.video_init_path
        init_frame = call_get_next_frame(init, i, video_init_path)
        print_init_frame_info(init_frame)
        self.args.args.init_image = init_frame
        self.args.args.init_image_box = None  # init_image_box not used in this case
        self.args.args.strength = max(0.0, min(1.0, step.step_data.strength))

    def _update_video_mask_for_current_frame(self, i):
        video_mask_path = self.args.anim_args.video_mask_path
        is_mask = True
        mask_init_frame = call_get_next_frame(self, i, video_mask_path, is_mask)
        new_mask = call_get_mask_from_file_with_frame(self, mask_init_frame)
        self.args.args.mask_file = new_mask
        self.args.root.noise_mask = new_mask
        mask.vals['video_mask'] = new_mask

    def update_video_data_for_current_frame(self, i, step):
        if self.animation_mode.has_video_input:
            self._update_video_input_for_current_frame(i, step)
        if self.args.anim_args.use_mask_video:
            self._update_video_mask_for_current_frame(i)

    def update_mask_image(self, step, mask):
        is_use_mask = self.args.args.use_mask
        if is_use_mask:
            has_sample = self.args.root.init_sample is not None
            if has_sample:
                mask_seq = step.schedule.mask_seq
                sample = self.args.root.init_sample
                self.args.args.mask_image = call_compose_mask_with_check(self, mask_seq, mask.vals, sample)
            else:
                self.args.args.mask_image = None  # we need it only after the first frame anyway

    def prepare_generation(self, data, step, i):
        # TODO move all of this to Step?
        if i > self.args.anim_args.max_frames - 1:
            return  # FIXME? sus
        self.update_some_args_for_current_step(step, i)
        self.update_seed_and_checkpoint_for_current_step(i)
        self.update_sub_seed_schedule_for_current_step(i)
        self.prompt_for_current_step(i)
        self.update_video_data_for_current_frame(i, step)
        self.update_mask_image(step, data.mask)
        self.animation_keys.update(i)
        opt_utils.setup(self, step.schedule)
        memory_utils.handle_vram_if_depth_is_predicted(data)

    @staticmethod
    def create_output_directory_for_the_batch(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Saving animation frames to:\n{directory}")

    @staticmethod
    def create_parseq_adapter(args):
        adapter = ParseqAdapter(args.parseq_args, args.anim_args, args.video_args, args.controlnet_args, args.loop_args)
        # Always enable pseudo-3d with parseq. No need for an extra toggle:
        # Whether it's used or not in practice is defined by the schedules
        if adapter.use_parseq:
            args.anim_args.flip_2d_perspective = True
        return adapter

    @staticmethod
    def init_looper_if_active(args, loop_args):
        if loop_args.use_looper:
            print("Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
        if args.strength == 0:
            raise RuntimeError("Strength needs to be greater than 0 in Init tab")
        args.strength_0_no_init = False
        args.seed_behavior = "schedule"
        if not isJson(loop_args.init_images):
            raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")

    @staticmethod
    def select_prompts(parseq_adapter, anim_args, animation_keys, root):
        return animation_keys.deform_keys.prompts if parseq_adapter.manages_prompts() \
            else RenderData.expand_prompts_out_to_per_frame(anim_args, root)

    @staticmethod
    def is_composite_with_depth_mask(anim_args):
        return anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth'

    @staticmethod
    def create_depth_model_and_enable_depth_map_saving_if_active(anim_mode, root, anim_args, args):
        # depth-based hybrid composite mask requires saved depth maps
        # TODO avoid or isolate side effect:
        anim_args.save_depth_maps = (anim_mode.is_predicting_depths
                                     and RenderData.is_composite_with_depth_mask(anim_args))
        return DepthModel(root.models_path,
                          memory_utils.select_depth_device(root),
                          root.half_precision,
                          keep_in_vram=anim_mode.is_keep_in_vram,
                          depth_algorithm=anim_args.depth_algorithm,
                          Width=args.W, Height=args.H,
                          midas_weight=anim_args.midas_weight) \
            if anim_mode.is_predicting_depths else None

    @staticmethod
    def expand_prompts_out_to_per_frame(anim_args, root):
        prompt_series = pd.Series([np.nan for _ in range(anim_args.max_frames)])
        for i, prompt in root.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        return prompt_series.ffill().bfill()

    @staticmethod
    def handle_controlnet_video_input_frames_generation(controlnet_args, args, anim_args):
        if is_controlnet_enabled(controlnet_args):
            unpack_controlnet_vids(args, anim_args, controlnet_args)

    @staticmethod
    def save_settings_txt(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root):
        save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)

    @staticmethod
    def maybe_resume_from_timestring(anim_args, root):
        root.timestring = anim_args.resume_timestring if anim_args.resume_from_timestring else root.timestring
