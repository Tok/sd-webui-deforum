from modules.shared import cmd_opts  # keep readonly


class MemoryUtils:
    # Don't put any variables here, it's meant for static methods only.

    @staticmethod
    def is_low_or_med_vram():
        # TODO Ideally this should only be called once at the beginning of the render.
        # Perhaps add a constant bool to RenderInit.
        return cmd_opts.lowvram or cmd_opts.medvram  # cmd_opts are imported from elsewhere. keep readonly

    @staticmethod
    def select_depth_device(root):
        return 'cpu' if MemoryUtils.is_low_or_med_vram() else root.device
