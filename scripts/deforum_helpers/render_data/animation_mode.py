import dataclasses
import os

from typing import Any
from ..generate import isJson
from ..hybrid_video import hybrid_generation


@dataclasses.dataclass
class AnimationMode:
    hybrid_input_files: Any = None
    hybrid_frame_path: str = None
    prev_flow: Any = None

    @staticmethod
    def _is_2d_or_3d_mode(anim_args):
        return anim_args.animation_mode in ['2D', '3D']

    @staticmethod
    def _is_using_hybris_frames(anim_args):
        return (anim_args.hybrid_composite != 'None'
                or anim_args.hybrid_motion in ['Affine', 'Perspective', 'Optical Flow'])

    @staticmethod
    def _is_needing_hybris_frames(anim_args):
        return AnimationMode._is_2d_or_3d_mode(anim_args) and AnimationMode._is_using_hybris_frames(anim_args)

    @staticmethod
    def from_args(anim_args, args, root):
        init_prev_flow = None
        init_hybrid_input_files: Any = None
        init_hybrid_frame_path = None
        if AnimationMode._is_needing_hybris_frames(anim_args):
            # handle hybrid video generation
            # hybrid_generation may cause side effects on args and anim_args
            _, __, init_hybrid_input_files = hybrid_generation(args, anim_args, root)
            # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
            init_hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')
        return AnimationMode(init_hybrid_input_files, init_hybrid_frame_path, init_prev_flow)
