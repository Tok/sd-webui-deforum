RED = "\033[31m"
ORANGE = "\033[38;5;208m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
BLUE = "\033[34m"
INDIGO = "\033[38;5;66m"
VIOLET = "\033[38;5;130m"
BLACK = "\033[30m"
WHITE = "\033[37m"

BOLD = "\033[1m"
UNDERLINE = "\033[4m"

RESET = "\033[0m"


def print_tween_frame_from_to_info(cadence, tween_values, start_i, end_i):
    print()  # additional newline to skip out of progress bar.
    if end_i > 0:
        formatted_values = [f"{val:.2f}" for val in tween_values]
        print(f"{ORANGE}Creating in-between: {RESET}{cadence} frames ({start_i}-->{end_i}){formatted_values}")


def print_animation_frame_info(init):
    print(f"{CYAN}Animation frame: {RESET}{init.indexes.frame.i}/{init.args.anim_args.max_frames}")


def print_tween_frame_info(data, indexes, cadence_flow, tween, if_disable=True):
    if not if_disable:
        msg_flow_name = '' if cadence_flow is None else data.args.anim_args.optical_flow_cadence + ' optical flow '
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


def print_warning_generate_returned_no_image():
    print(f"{YELLOW}Warning: {RESET}Generate returned no image. Skipping to next iteration.")
