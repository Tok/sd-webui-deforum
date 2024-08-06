import os

from . import filename_utils, memory_utils


def generate_and_save_depth_map_if_active(data, opencv_image):
    if data.args.anim_args.save_depth_maps:
        memory_utils.handle_vram_before_depth_map_generation(data)
        depth = data.depth_model.predict(opencv_image, data.args.anim_args.midas_weight,
                                         data.args.root.half_precision)
        depth_filename = filename_utils.depth_frame(data, data.indexes)
        data.depth_model.save(os.path.join(data.output_directory, depth_filename), depth)
        memory_utils.handle_vram_after_depth_map_generation(data)
        return depth
