

def print_animation_frame_info(init, indexes):
    print(f"\033[36mAnimation frame: \033[0m{indexes.frame.i}/{init.args.anim_args.max_frames}")


def print_tween_frame_info(init, indexes, cadence_flow, tween):
    msg_flow_name = '' if cadence_flow is None else init.args.anim_args.optical_flow_cadence + ' optical flow '
    msg_frame_info = f"cadence frame: {indexes.tween.i}; tween: {tween:0.2f};"
    print(f"Creating in-between {msg_flow_name}{msg_frame_info}")


def print_init_frame_info(init_frame):
    print(f"Using video init frame {init_frame}")


def print_optical_flow_info(init, optical_flow_redo_generation):
    msg_start = "Optical flow redo is diffusing and warping using"
    msg_end = "optical flow before generation."
    print(f"{msg_start} {optical_flow_redo_generation} and seed {init.args.args.seed} {msg_end}")


def print_redo_generation_info(init, n):
    print(f"Redo generation {n + 1} of {int(init.args.anim_args.diffusion_redo)} before final generation")
