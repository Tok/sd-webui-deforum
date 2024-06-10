import os

import cv2

from .filename_utils import tween_frame_name
from ...masks import do_overlay_mask


def force_tween_to_grayscale_if_required(init, image):
    if init.args.anim_args.color_force_grayscale:
        gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        return image


def force_to_grayscale_if_required(init, image):
    if init.args.anim_args.color_force_grayscale:
        gray_image = ImageOps.grayscale(image)
        return ImageOps.colorize(gray_image, black="black", white="white")
    else:
        return image


def add_overlay_mask_if_active(init, image, is_tween: bool = False):
    is_use_overlay = init.args.args.overlay_mask
    is_use_mask = init.args.anim_args.use_mask_video or init.args.args.use_mask
    if is_use_overlay and is_use_mask:
        index = indexes.tween.i if is_tween else indexes.frame.i
        is_bgr_array = True
        return do_overlay_mask(init.args.args, init.args.anim_args, image, index, is_bgr_array)
    else:
        return image


def save_cadence_frame(init, indexes, image):
    filename = tween_frame_name(init, indexes)
    save_path: str = os.path.join(init.args.args.outdir, filename)
    cv2.imwrite(save_path, image)
