from ...animation import anim_frame_warp


def put_if_present(dictionary, key, value):
    # FIXME does this even make sense? (..reading py docs)
    if value is not None:
        dictionary[key] = value


def call_anim_frame_warp(init, image, frame_index, depth):
    return anim_frame_warp(image,
                           init.args.args,
                           init.args.anim_args,
                           init.animation_keys.deform_keys,
                           frame_index,
                           init.depth_model,
                           depth=depth,
                           device=init.args.root.device,
                           half_precision=init.args.root.half_precision)
