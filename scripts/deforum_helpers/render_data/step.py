import dataclasses

from typing import Any
from . import AnimationKeys, AnimationMode, Srt
from ..parseq_adapter import ParseqAdapter


@dataclasses.dataclass
class StepArgs:
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
        return StepArgs(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)


@dataclasses.dataclass
class Step:
    seed: int
    step_args: StepArgs
    parseq_adapter: Any
    srt: Any
    animation_keys: AnimationKeys
    animation_mode: AnimationMode

    def __new__(cls, *args, **kwargs):
        raise TypeError("Use Step.create() to create new instances.")

    @classmethod
    def create_parseq_adapter(cls, args):
        adapter = ParseqAdapter(args.parseq_args, args.anim_args, args.video_args, args.controlnet_args, args.loop_args)
        # Always enable pseudo-3d with parseq. No need for an extra toggle:
        # Whether it's used or not in practice is defined by the schedules
        if adapter.use_parseq:
            args.anim_args.flip_2d_perspective = True
        return adapter

    @classmethod
    def create(cls, args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root) -> 'Step':
        step_args = StepArgs(args, parseq_args, anim_args, video_args, controlnet_args, loop_args, opts, root)
        parseq_adapter = Step.create_parseq_adapter(step_args)
        instance = object.__new__(cls)  # creating the instance without raising the type error defined in __new__.
        instance.__init__(args.seed, step_args, parseq_adapter,
            Srt.create_if_active(opts.data, args.outdir, root.timestring, video_args.fps),
            AnimationKeys.from_args(step_args, parseq_adapter, args.seed),
            AnimationMode.from_args(step_args))  # possible side effects on anim_args and args
        return instance
