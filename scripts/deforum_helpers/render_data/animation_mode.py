import dataclasses
import os

from typing import Any
from ..hybrid_video import hybrid_generation


@dataclasses.dataclass
class AnimationMode:
    has_video_input: bool = False
    hybrid_input_files: Any = None
    hybrid_frame_path: str = None
    prev_flow: Any = None
    is_predicting_depths: Any = None


    @staticmethod
    def _has_video_input(anim_args) -> bool:
        return AnimationMode._is_2d_or_3d_mode(anim_args) and AnimationMode._is_using_hybris_frames(anim_args)


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
    def _is_load_depth_model_for_3d(args, anim_args):
        is_depth_warped_3d = anim_args.animation_mode == '3D' and anim_args.use_depth_warping
        is_composite_with_depth = anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth']
        is_depth_used = is_depth_warped_3d or anim_args.save_depth_maps or is_composite_with_depth
        return is_depth_used and not args.motion_preview_mode

    @staticmethod
    def from_args(anim_args, args, root):
        init_hybrid_input_files: Any = None
        if AnimationMode._is_needing_hybris_frames(anim_args):
            # handle hybrid video generation
            # hybrid_generation may cause side effects on args and anim_args
            _, __, init_hybrid_input_files = hybrid_generation(args, anim_args, root)
            # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
            init_hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')

        return AnimationMode(AnimationMode._has_video_input(anim_args),
                             init_hybrid_input_files,
                             None,
                             None,
                             AnimationMode._is_load_depth_model_for_3d(args, anim_args))
