from .data.step.step import Step
from .util.call.hybrid import call_get_flow_from_images, call_hybrid_composite
from .util.fun_utils import tube

"""
This module provides functions for processing images through various transformations.
The `tube` function allows chaining these transformations together to create flexible image processing pipelines.
Easily experiment by changing, or changing the order of function calls in the tube.

All functions within the tube take and return an image (`img` argument). They may (and must) pass through
the original image unchanged if a specific transformation is disabled or not required.

Example:
transformed_image = my_tube(arguments)(original_image)
"""


def frame_transformation_tube(init, indexes, step, images):
    # make sure `img` stays the last argument in each call.
    return tube(lambda img: step.apply_frame_warp_transform(init, indexes, img),
                lambda img: step.do_hybrid_compositing_before_motion(init, indexes, img),
                lambda img: Step.apply_hybrid_motion_ransac_transform(init, indexes, images, img),
                lambda img: Step.apply_hybrid_motion_optical_flow(init, indexes, images, img),
                lambda img: step.do_normal_hybrid_compositing_after_motion(init, indexes, img),
                lambda img: Step.apply_color_matching(init, images, img),
                lambda img: Step.transform_to_grayscale_if_active(init, images, img))


def contrast_transformation_tube(init, step, mask):
    return tube(lambda img: step.apply_scaling(img),
                lambda img: step.apply_anti_blur(init, mask, img))


def noise_transformation_tube(init, step):
    return tube(lambda img: step.apply_frame_noising(init, step, img))


def optical_flow_redo_tube(init, optical_flow, images):
    return tube(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                lambda img: image_transform_optical_flow(  # TODO create img.get_flow
                    img, call_get_flow_from_images(init, images.previous, img, optical_flow),
                    step.init.redo_flow_factor))


def hybrid_video_after_generation_tube(init, indexes, step):
    return tube(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                lambda img: call_hybrid_composite(init, indexes.frame.i, img, step.init.hybrid_comp_schedules),
                lambda img: Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


def color_match_tube(init, images):
    return tube(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                lambda img: maintain_colors(img, images.color_match, init.args.anim_args.color_coherence),
                lambda img: Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
