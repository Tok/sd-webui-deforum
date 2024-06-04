import numexpr
import numpy as np
import os
import pandas as pd

from dataclasses import dataclass
from typing import Any
from .data.anim import AnimationKeys, AnimationMode
from .data.subtitle import Srt
from .util import MemoryUtils
from ..deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from ..depth import DepthModel
from ..generate import isJson
from ..parseq_adapter import ParseqAdapter
from ..settings import save_settings_from_animation_run


@dataclass(init=True, frozen=True, repr=False, eq=False)
class RenderInitArgs:
    # TODO eventually this should only keep the information required to run render_animation once
    #  for now it's just a direct reference or a copy of the actual args provided to the render_animation call.
    args: Any = None
    parseq_args: Any = None
    anim_args: Any = None
    video_args: Any = None
    controlnet_args: Any = None
    loop_args: Any = None
    opts: Any = None
    root: Any = None

    @classmethod
    def create(cls, args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root):
        return RenderInitArgs(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)


@dataclass(init=True, frozen=True, repr=False, eq=False)
class RenderInit:
    """The purpose of this class is to group and control all data used in render_animation"""
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

    def __new__(cls, *args, **kwargs):
        raise TypeError("Use RenderInit.create() to create new instances.")

    def is_3d(self):
        return self.args.anim_args.animation_mode == '3D'

    def is_3d_with_med_or_low_vram(self):
        return self.is_3d() and MemoryUtils.is_low_or_med_vram()

    def width(self) -> int:
        return self.args.args.W

    def height(self) -> int:
        return self.args.args.H

    def dimensions(self) -> tuple[int, int]:
        # TODO should ideally only be called once each render
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

    @classmethod
    def create_output_directory_for_the_batch(cls, directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Saving animation frames to:\n{directory}")

    @classmethod
    def create_parseq_adapter(cls, args):
        adapter = ParseqAdapter(args.parseq_args, args.anim_args, args.video_args, args.controlnet_args, args.loop_args)
        # Always enable pseudo-3d with parseq. No need for an extra toggle:
        # Whether it's used or not in practice is defined by the schedules
        if adapter.use_parseq:
            args.anim_args.flip_2d_perspective = True
        return adapter

    @classmethod
    def init_looper_if_active(cls, args, loop_args):
        if loop_args.use_looper:
            print("Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
        if args.strength == 0:
            raise RuntimeError("Strength needs to be greater than 0 in Init tab")
        args.strength_0_no_init = False
        args.seed_behavior = "schedule"
        if not isJson(loop_args.init_images):
            raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")

    @classmethod
    def select_prompts(cls, parseq_adapter, anim_args, animation_keys, root):
        return animation_keys.deform_keys.prompts if parseq_adapter.manages_prompts() \
            else RenderInit.expand_prompts_out_to_per_frame(anim_args, root)

    @classmethod
    def is_composite_with_depth_mask(cls, anim_args):
        return anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth'

    @classmethod
    def create_depth_model_and_enable_depth_map_saving_if_active(cls, anim_mode, root, anim_args, args):
        # depth-based hybrid composite mask requires saved depth maps
        # TODO avoid or isolate side effect:
        anim_args.save_depth_maps = (anim_mode.is_predicting_depths
                                     and RenderInit.is_composite_with_depth_mask(anim_args))
        return DepthModel(root.models_path,
                          MemoryUtils.select_depth_device(root),
                          root.half_precision,
                          keep_in_vram=anim_mode.is_keep_in_vram,
                          depth_algorithm=anim_args.depth_algorithm,
                          Width=args.W, Height=args.H,
                          midas_weight=anim_args.midas_weight) \
            if anim_mode.is_predicting_depths else None

    @classmethod
    def expand_prompts_out_to_per_frame(cls, anim_args, root):
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
        for i, prompt in root.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        return prompt_series.ffill().bfill()

    @classmethod
    def handle_controlnet_video_input_frames_generation(cls, controlnet_args, args, anim_args):
        if is_controlnet_enabled(controlnet_args):
            unpack_controlnet_vids(args, anim_args, controlnet_args)

    @classmethod
    def save_settings_txt(cls, args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root):
        save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)

    @classmethod
    def maybe_resume_from_timestring(cls, anim_args, root):
        root.timestring = anim_args.resume_timestring if anim_args.resume_from_timestring else root.timestring

    @classmethod
    def do_void_inits(cls, args, loop_args, controlnet_args, anim_args, parseq_args, video_args, root):
        # TODO all of those calls may cause a change in on of the passed args.
        # Ideally it may be refactored so each one returns a new instance of the potentially changed args that are then
        # attached as a property to this class to be used for one single render only.
        RenderInit.init_looper_if_active(args, loop_args)
        RenderInit.handle_controlnet_video_input_frames_generation(controlnet_args, args, anim_args)
        RenderInit.create_output_directory_for_the_batch(args.outdir)
        RenderInit.save_settings_txt(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)
        RenderInit.maybe_resume_from_timestring(anim_args, root)

    @classmethod
    def create(cls, args_argument, parseq_args, anim_args, video_args, controlnet_args,
               loop_args, opts, root) -> 'RenderInit':
        # TODO deepcopy args?
        args = RenderInitArgs(args_argument, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
        output_directory = args_argument.outdir
        is_use_mask = args_argument.use_mask
        parseq_adapter = RenderInit.create_parseq_adapter(args)
        srt = Srt.create_if_active(opts.data, output_directory, root.timestring, video_args.fps)
        animation_keys = AnimationKeys.from_args(args, parseq_adapter, args_argument.seed)
        animation_mode = AnimationMode.from_args(args)
        prompt_series = RenderInit.select_prompts(parseq_adapter, anim_args, animation_keys, root)
        depth_model = RenderInit.create_depth_model_and_enable_depth_map_saving_if_active(
            animation_mode, root, anim_args, args_argument)
        instance = object.__new__(cls)  # creating the instance without raising the type error defined in __new__.
        instance.__init__(args_argument.seed, args, parseq_adapter, srt,
                          animation_keys, animation_mode, prompt_series,
                          depth_model, output_directory, is_use_mask)
        # Ideally, a call to render_animation in render.py shouldn't cause changes in any of the args passed there.
        # It may be preferable to work on temporary copies within tight scope.
        # TODO avoid or isolate more side effects
        RenderInit.do_void_inits(args_argument, loop_args, controlnet_args, anim_args, parseq_args, video_args, root)

        return instance
