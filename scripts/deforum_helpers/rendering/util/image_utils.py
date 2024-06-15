import os

import cv2

from .filename_utils import tween_frame_name


def save_cadence_frame(init, indexes, image):
    filename = tween_frame_name(init, indexes)
    save_path: str = os.path.join(init.args.args.outdir, filename)
    cv2.imwrite(save_path, image)


def save_cadence_frame_and_depth_map_if_active(init, indexes, image):
    save_cadence_frame(init, indexes, image)
    if init.args.anim_args.save_depth_maps:
        dm_save_path = os.path.join(init.output_directory, filename_utils.tween_depth_frame(init, indexes))
        init.depth_model.save(dm_save_path, step.depth)


def save_and_return_frame(init, indexes, image):
    save_cadence_frame_and_depth_map_if_active(init, indexes, image)
    return image
