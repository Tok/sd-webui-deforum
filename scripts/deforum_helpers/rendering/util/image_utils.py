import os

import cv2

from .filename_utils import tween_frame_name
from ..data.render_data import RenderData


def save_cadence_frame(data: RenderData, indexes, image):
    filename = tween_frame_name(data, indexes)
    save_path: str = os.path.join(data.args.args.outdir, filename)
    cv2.imwrite(save_path, image)


def save_cadence_frame_and_depth_map_if_active(data: RenderData, indexes, image):
    save_cadence_frame(data, indexes, image)
    if data.args.anim_args.save_depth_maps:
        dm_save_path = os.path.join(data.output_directory, filename_utils.tween_depth_frame(data, indexes))
        data.depth_model.save(dm_save_path, step.depth)


def save_and_return_frame(data: RenderData, indexes, image):
    save_cadence_frame_and_depth_map_if_active(data, indexes, image)
    return image
