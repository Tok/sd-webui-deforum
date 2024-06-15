from .utils import put_if_present


def setup(init, schedule):
    data = init.args.opts.data
    if init.has_img2img_fix_steps():
        # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
        data["img2img_fix_steps"] = False
    put_if_present(data, "CLIP_stop_at_last_layers", schedule.clipskip)
    put_if_present(data, "initial_noise_multiplier", schedule.noise_multiplier)
    put_if_present(data, "eta_ddim", schedule.eta_ddim)
    put_if_present(data, "eta_ancestral", schedule.eta_ancestral)


def generation_info_for_subtitles(render_data):
    return render_data.args.opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])


def is_generate_subtitles(render_data):
    return render_data.args.opts.data.get("deforum_save_gen_info_as_srt")

