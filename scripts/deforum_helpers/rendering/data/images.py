from dataclasses import dataclass

import cv2
from cv2.typing import MatLike


@dataclass(init=True, frozen=False, repr=False, eq=True)
class Images:
    previous: MatLike | None = None
    color_match: MatLike = None

    def has_previous(self):
        return self.previous is not None

    @staticmethod
    def _load_color_match_sample(init) -> MatLike:
        """get color match for 'Image' color coherence only once, before loop"""
        if init.args.anim_args.color_coherence == 'Image':
            raw_image = load_image(init.args.anim_args.color_coherence_image_path, None)
            resized = raw_image.resize(init.dimensions(), PIL.Image.LANCZOS)
            return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

    @staticmethod
    def create(init):
        return Images(None, Images._load_color_match_sample(init))
