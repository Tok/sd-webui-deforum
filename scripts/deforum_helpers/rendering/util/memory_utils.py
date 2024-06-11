# noinspection PyUnresolvedReferences
from modules import lowvram, devices, sd_hijack
# noinspection PyUnresolvedReferences
from modules.shared import cmd_opts  # keep readonly


def is_low_or_med_vram():
    # TODO Ideally this should only be called once at the beginning of the render.
    # Perhaps add a constant bool to RenderInit.
    return cmd_opts.lowvram or cmd_opts.medvram  # cmd_opts are imported from elsewhere. keep readonly


def handle_med_or_low_vram_before_step(init):
    if init.is_3d_with_med_or_low_vram():
        # Unload the main checkpoint and load the depth model
        lowvram.send_everything_to_cpu()
        sd_hijack.model_hijack.undo_hijack(sd_model)
        devices.torch_gc()
        if init.animation_mode.is_predicting_depths:
            init.animation_mode.depth_model.to(init.args.root.device)


def handle_vram_if_depth_is_predicted(init):
    if init.animation_mode.is_predicting_depths:
        if init.is_3d_with_med_or_low_vram():
            init.depth_model.to('cpu')
            devices.torch_gc()
            lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            sd_hijack.model_hijack.hijack(sd_model)


def handle_vram_before_depth_map_generation(init):
    if is_low_or_med_vram():
        lowvram.send_everything_to_cpu()
        sd_hijack.model_hijack.undo_hijack(sd_model)
        devices.torch_gc()
        init.depth_model.to(init.args.root.device)


def handle_vram_after_depth_map_generation(init):
    if is_low_or_med_vram():
        init.depth_model.to('cpu')
        devices.torch_gc()
        lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
        sd_hijack.model_hijack.hijack(sd_model)


def select_depth_device(root):
    return 'cpu' if is_low_or_med_vram() else root.device


def keep_3d_models_in_vram(step_args):
    return step_args.opts.data.get("deforum_keep_3d_models_in_vram", False)
