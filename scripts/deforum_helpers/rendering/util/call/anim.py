from ..utils import context
from ....animation import anim_frame_warp


def call_anim_frame_warp(init, i, image, depth):
    with context(init.args) as ia:
        return anim_frame_warp(image, ia.args, ia.anim_args, init.animation_keys.deform_keys, i, init.depth_model,
                               depth=depth, device=ia.root.device, half_precision=ia.root.half_precision)
