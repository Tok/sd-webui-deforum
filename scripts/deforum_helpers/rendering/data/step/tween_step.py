from dataclasses import dataclass
from typing import Any

from ...util.image_utils import add_overlay_mask_if_active, force_tween_to_grayscale_if_required


@dataclass(init=True, frozen=False, repr=False, eq=False)
class TweenStep:
    """cadence vars"""
    tween: float
    cadence_flow: Any
    cadence_flow_inc: Any

    @staticmethod
    def create(indexes):
        from_i = indexes.frame.i - indexes.tween.start
        to_i = indexes.tween.i - indexes.tween.start + 1
        tween = float(to_i) / float(from_i)
        return TweenStep(tween, None, None)

    def generate_tween_image(self, init, indexes, step, turbo):
        is_tween = True
        warped = turbo.do_optical_flow_cadence_after_animation_warping(init, indexes, step, self)
        recolored = force_tween_to_grayscale_if_required(init, warped)
        masked = add_overlay_mask_if_active(init, recolored, is_tween)
        return masked
