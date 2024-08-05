# noinspection PyUnresolvedReferences
from modules.shared import opts

COLOUR_RGB = '\x1b[38;2;%d;%d;%dm'
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

RESET = "\x1b[0m"


def is_verbose():
    """Checks if extra console output is enabled in deforum settings."""
    return opts.data.get("deforum_debug_mode_enabled", False)


def clear_previous_line():
    print("\033[F\033[K", end="")  # "\033[" is the ANSI escape sequence, "F" is cursor up, "K" is clear line.


def print_tween_frame_from_to_info(key_step, is_disabled=True):
    if not is_disabled:  # replaced with prog bar, but value info print may be useful
        tween_values = key_step.tween_values
        start_i = key_step.tweens[0].i()
        end_i = key_step.tweens[-1].i()
        if end_i > 0:
            formatted_values = [f"{val:.2f}" for val in tween_values]
            count = end_i - start_i + 1
            print(f"{ORANGE}Creating in-between: {RESET}{count} frames ({start_i}-->{end_i}){formatted_values}")


def print_animation_frame_info(i, max_frames):
    print("")
    print(f"{CYAN}Animation frame: {RESET}{i}/{max_frames}")


def print_tween_frame_info(data, indexes, cadence_flow, tween, is_disabled=True):
    if not is_disabled:  # disabled because it's spamming the cli on high cadence settings.
        msg_flow_name = '' if cadence_flow is None else data.args.anim_args.optical_flow_cadence + ' optical flow '
        msg_frame_info = f"cadence frame: {indexes.tween.i}; tween: {tween:0.2f};"
        print(f"Creating in-between {msg_flow_name}{msg_frame_info}")


def print_init_frame_info(init_frame):
    print(f"Using video init frame {init_frame}")


def print_optical_flow_info(data, optical_flow_redo_generation):
    msg_start = "Optical flow redo is diffusing and warping using"
    msg_end = "optical flow before generation."
    print(f"{msg_start} {optical_flow_redo_generation} and seed {data.args.args.seed} {msg_end}")


def print_redo_generation_info(data, n):
    print(f"Redo generation {n + 1} of {int(data.args.anim_args.diffusion_redo)} before final generation")


def print_tween_step_creation_info(key_steps, index_dist):
    tween_count = sum(len(ks.tweens) for ks in key_steps)
    msg_start = f"Created {len(key_steps)} key frames with {tween_count} tweens."
    msg_end = f"Key frame index distribution: '{index_dist.name}'."
    info(f"{msg_start} {msg_end}")


def print_key_step_debug_info_if_verbose(key_steps):
    for i, ks in enumerate(key_steps):
        tween_indices = [t.i() for t in ks.tweens]
        debug(f"Key frame {ks.i} has {len(tween_indices)} tweens: {tween_indices}")


def print_warning_generate_returned_no_image():
    print(f"{YELLOW}Warning: {RESET}Generate returned no image. Skipping to next iteration.")


def print_cuda_memory_state(cuda):
    for i in range(cuda.device_count()):
        print(f"CUDA memory allocated on device {i}: {cuda.memory_allocated(i)} of {cuda.max_memory_allocated(i)}")
        print(f"CUDA memory reserved on device {i}: {cuda.memory_reserved(i)} of {cuda.max_memory_reserved(i)}")


def info(s: str):
    print(f"Info: {s}")


def warn(s: str):
    print(f"{ORANGE}Warn: {RESET}{s}")


def debug(s: str):
    if is_verbose():
        eye_catcher = "###"
        print(f"{YELLOW}{BOLD}{eye_catcher} Debug: {RESET}{s}")
