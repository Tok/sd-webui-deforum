from ....subtitle_handler import format_animation_params, write_frame_subtitle


def call_format_animation_params(init, i, params_to_print):
    return format_animation_params(init.animation_keys.deform_keys, init.prompt_series, i, params_to_print)


def call_write_frame_subtitle(init, i, params_string, is_cadence: bool = False) -> None:
    text = f"F#: {i}; Cadence: {is_cadence}; Seed: {init.args.args.seed}; {params_string}"
    write_frame_subtitle(init.srt.filename, i, init.srt.frame_duration, text)
