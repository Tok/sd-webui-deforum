from modules.shared import cmd_opts  # keep readonly


class MemoryUtils():
    # Don't put any variables here, it's meant for static methods only.

    @staticmethod
    def is_low_or_med_vram():
        return cmd_opts.lowvram or cmd_opts.medvram  # cmd_opts are imported from elsewhere. keep readonly

    @staticmethod
    def select_depth_device(root):
        return 'cpu' if MemoryUtils.is_low_or_med_vram() else root.device
