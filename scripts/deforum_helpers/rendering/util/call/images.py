from ....load_images import get_mask_from_file


def call_get_mask_from_file(init, i, is_mask: bool = False):
    next_frame = get_next_frame(init.output_directory, init.args.anim_args.video_mask_path, i, is_mask)
    return get_mask_from_file(next_frame, init.args.args)


def call_get_mask_from_file_with_frame(init, frame):
    return get_mask_from_file(frame, init.args.args)
