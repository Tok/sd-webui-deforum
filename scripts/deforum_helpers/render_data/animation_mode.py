import dataclasses
import os

from typing import Any
from ..hybrid_video import hybrid_generation
from ..RAFT import RAFT


@dataclasses.dataclass
class AnimationMode:
    has_video_input: bool = False
    hybrid_input_files: Any = None
    hybrid_frame_path: str = ""
    prev_flow: Any = None
    is_keep_in_vram: bool = False
    depth_model: Any = None
    raft_model: Any = None

    def is_predicting_depths(self) -> bool:
        return self.depth_model is not None

    def is_raft_active(self) -> bool:
        return self.raft_model is not None

    def cleanup(self):
        if self.is_predicting_depths() and not self.is_keep_in_vram:
            self.depth_model.delete_model()  # handles adabins too
        if self.is_raft_active():
            self.raft_model.delete_model()


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
    def _load_raft_if_active(anim_args, args):
        is_cadenced_raft = anim_args.optical_flow_cadence == "RAFT" and int(anim_args.diffusion_cadence) > 1
        is_optical_flow_raft = anim_args.hybrid_motion == "Optical Flow" and anim_args.hybrid_flow_method == "RAFT"
        is_raft_redo = anim_args.optical_flow_redo_generation == "RAFT"
        is_load_raft = (is_cadenced_raft or is_optical_flow_raft or is_raft_redo) and not args.motion_preview_mode
        if is_load_raft:
            print("Loading RAFT model...")
        return RAFT() if is_load_raft else None

    @staticmethod
    def _load_depth_model_if_active(args, anim_args, opts):
        return AnimationMode._is_load_depth_model_for_3d(args, anim_args) \
            if opts.data.get("deforum_keep_3d_models_in_vram", False) else None

    @staticmethod
    def from_args(anim_args, args, opts, root):
        init_hybrid_input_files: Any = None
        init_hybrid_frame_path = ""
        if AnimationMode._is_needing_hybris_frames(anim_args):
            # handle hybrid video generation
            # hybrid_generation may cause side effects on args and anim_args
            _, __, init_hybrid_input_files = hybrid_generation(args, anim_args, root)
            # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
            init_hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')
        return AnimationMode(AnimationMode._has_video_input(anim_args),
                             init_hybrid_input_files,
                             init_hybrid_frame_path,
                             None,
                             opts.data.get("deforum_keep_3d_models_in_vram", False),
                             AnimationMode._load_depth_model_if_active(args, anim_args, opts),
                             AnimationMode._load_raft_if_active(anim_args, args))
