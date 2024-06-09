import os
from dataclasses import dataclass
from typing import Any

import cv2
import numexpr
import numpy as np
import pandas as pd
from PIL import Image

from .anim import AnimationKeys, AnimationMode
from .subtitle import Srt
from ..util import memory_utils
from ..util.utils import context
from ...args import RootArgs
from ...deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from ...depth import DepthModel
from ...generate import (isJson)
from ...parseq_adapter import ParseqAdapter
from ...settings import save_settings_from_animation_run


@dataclass(init=True, frozen=True, repr=False, eq=False)
class RenderInitArgs:
    args: Any = None
    parseq_args: Any = None
    anim_args: Any = None
    video_args: Any = None
    controlnet_args: Any = None
    loop_args: Any = None
    opts: Any = None
    root: Any = None

    @staticmethod
    def create(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root):
        return RenderInitArgs(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)


@dataclass(init=True, frozen=True, repr=False, eq=False)
class RenderInit:
    """The purpose of this class is to group and control all data used in render_animation"""
    root: RootArgs
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

    def is_color_match_to_be_initialized(self, color_match_sample):
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

    def update_some_args_for_current_progression_step(self, step, noised_image):
        # use transformed previous frame as init for current
        self.args.args.use_init = True
        self.root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
        self.args.args.strength = max(0.0, min(1.0, step.init.strength))

    def update_some_args_for_current_step(self, indexes, step):
        i = indexes.frame.i
        keys = self.animation_keys.deform_keys
        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        self.args.args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[i])
        self.args.args.prompt = self.prompt_series[i]  # grab prompt for current frame
        self.args.args.scale = step.init.scale

    def update_seed_and_checkpoint_for_current_step(self, indexes):
        i = indexes.frame.i
        keys = self.animation_keys.deform_keys
        is_seed_scheduled = self.args.args.seed_behavior == 'schedule'
        is_seed_managed = self.parseq_adapter.manages_seed()
        is_seed_scheduled_or_managed = is_seed_scheduled or is_seed_managed
        if is_seed_scheduled_or_managed:
            self.args.args.seed = int(keys.seed_schedule_series[i])
        self.args.args.checkpoint = keys.checkpoint_schedule_series[i] \
            if self.args.anim_args.enable_checkpoint_scheduling else None

    def update_sub_seed_schedule_for_current_step(self, indexes):
        i = indexes.frame.i
        keys = self.animation_keys.deform_keys
        is_subseed_scheduling_enabled = self.args.anim_args.enable_subseed_scheduling
        is_seed_managed_by_parseq = self.parseq_adapter.manages_seed()
        if is_subseed_scheduling_enabled or is_seed_managed_by_parseq:
            self.root.subseed = int(keys.subseed_schedule_series[i])
        if is_subseed_scheduling_enabled and not is_seed_managed_by_parseq:
            self.root.subseed_strength = float(keys.subseed_strength_schedule_series[i])
        if is_seed_managed_by_parseq:
            self.root.subseed_strength = keys.subseed_strength_schedule_series[i]  # TODO not sure why not type-coerced.
            self.args.anim_args.enable_subseed_scheduling = True  # TODO should be enforced in init, not here.

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
            else RenderInit.expand_prompts_out_to_per_frame(anim_args, root)

    @staticmethod
    def is_composite_with_depth_mask(anim_args):
        return anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth'

    @staticmethod
    def create_depth_model_and_enable_depth_map_saving_if_active(anim_mode, root, anim_args, args):
        # depth-based hybrid composite mask requires saved depth maps
        # TODO avoid or isolate side effect:
        anim_args.save_depth_maps = (anim_mode.is_predicting_depths
                                     and RenderInit.is_composite_with_depth_mask(anim_args))
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
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
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

    @staticmethod
    def create(args, parseq_args, anim_args, video_args, controlnet_args,
               loop_args, opts, root) -> 'RenderInit':
        ri_args = RenderInitArgs(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
        output_directory = args.outdir
        is_use_mask = args.use_mask
        with context(RenderInit) as RI:
            parseq_adapter = RI.create_parseq_adapter(ri_args)
            srt = Srt.create_if_active(opts.data, output_directory, root.timestring, video_args.fps)
            animation_keys = AnimationKeys.from_args(ri_args, parseq_adapter, args.seed)
            animation_mode = AnimationMode.from_args(ri_args)
            prompt_series = RI.select_prompts(parseq_adapter, anim_args, animation_keys, root)
            depth_model = RI.create_depth_model_and_enable_depth_map_saving_if_active(
                animation_mode, root, anim_args, args)
            instance = RenderInit(root, args.seed, ri_args, parseq_adapter, srt, animation_keys,
                                  animation_mode, prompt_series, depth_model, output_directory, is_use_mask)
            RI.init_looper_if_active(args, loop_args)
            RI.handle_controlnet_video_input_frames_generation(controlnet_args, args, anim_args)
            RI.create_output_directory_for_the_batch(args.outdir)
            RI.save_settings_txt(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)
            RI.maybe_resume_from_timestring(anim_args, root)
            return instance
