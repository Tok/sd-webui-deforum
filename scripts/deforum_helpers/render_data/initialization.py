import dataclasses
import numexpr
import numpy as np
import os
import pandas as pd

from typing import Any
from modules.shared import cmd_opts  # keep readonly

from .anim import AnimationKeys, AnimationMode
from .subtitle import Srt
from ..deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from ..depth import DepthModel
from ..generate import isJson
from ..parseq_adapter import ParseqAdapter
from ..settings import save_settings_from_animation_run


@dataclasses.dataclass
class RenderInitArgs:
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


@dataclasses.dataclass
class RenderInit:
    """The purpose of this class is to group and control all data used in a single iteration of render_animation"""
    seed: int
    step_args: RenderInitArgs
    parseq_adapter: Any
    srt: Any
    animation_keys: AnimationKeys
    animation_mode: AnimationMode
    prompt_series: Any
    depth_model: Any

    def __new__(cls, *args, **kwargs):
        raise TypeError("Use RenderInit.create() to create new instances.")

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
    def is_low_or_med_vram(cls):
        # TODO move methods like this to a new static helper class or something
        return cmd_opts.lowvram or cmd_opts.medvram  # cmd_opts are imported from elsewhere. keep readonly

    @classmethod
    def select_depth_device(cls, root):
        return 'cpu' if RenderInit.is_low_or_med_vram() else root.device

    @classmethod
    def create_depth_model_and_enable_depth_map_saving_if_active(cls, anim_mode, root, anim_args, args):
        # depth-based hybrid composite mask requires saved depth maps
        # TODO avoid or isolate side effect:
        anim_args.save_depth_maps = (anim_mode.is_predicting_depths
                                     and RenderInit.is_composite_with_depth_mask(anim_args))
        return DepthModel(root.models_path,
                          RenderInit.select_depth_device(root),
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
    def create_output_directory_for_the_batch(cls, args):
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Saving animation frames to:\n{args.outdir}")

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
        RenderInit.create_output_directory_for_the_batch(args)
        RenderInit.save_settings_txt(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)
        RenderInit.maybe_resume_from_timestring(anim_args, root)

    @classmethod
    def create(cls, args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root) -> 'RenderInit':
        step_args = RenderInitArgs(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
        parseq_adapter = RenderInit.create_parseq_adapter(step_args)
        srt = Srt.create_if_active(opts.data, args.outdir, root.timestring, video_args.fps)
        animation_keys = AnimationKeys.from_args(step_args, parseq_adapter, args.seed)
        animation_mode = AnimationMode.from_args(step_args)
        prompt_series = RenderInit.select_prompts(parseq_adapter, anim_args, animation_keys, root)
        depth_model = RenderInit.create_depth_model_and_enable_depth_map_saving_if_active(animation_mode, root, anim_args, args)
        instance = object.__new__(cls)  # creating the instance without raising the type error defined in __new__.
        instance.__init__(args.seed, step_args, parseq_adapter, srt,
                          animation_keys, animation_mode, prompt_series, depth_model)
        # Ideally, a call to render_animation in render.py shouldn't cause changes in any of the args passed there.
        # It may be preferable to work on temporary copies within tight scope.
        # TODO avoid or isolate more side effects
        RenderInit.do_void_inits(args, loop_args, controlnet_args, anim_args, parseq_args, video_args, root)

        return instance
