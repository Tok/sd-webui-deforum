import os

from ..subtitle_handler import init_srt_file


class Srt:
    def __init__(self, filename: str, frame_duration: int):
        self._filename = filename
        self._frame_duration = frame_duration

    def __new__(cls, *args, **kwargs):  # locks the constructor to enforce proper initialization
        raise TypeError("Use Srt.create_if_active() to create new instances.")

    @classmethod
    def create_if_active(cls, opts_data, outdir: str, timestring: str, fps: float) -> 'Srt | None':
        if not opts_data.get("deforum_save_gen_info_as_srt", False):
            return None
        else:
            # create .srt file and set timeframe mechanism using FPS
            filename = os.path.join(outdir, f"{timestring}.srt")
            frame_duration = init_srt_file(filename, fps)

            instance = object.__new__(cls)  # creating the instance without raising the type error defined in __new__.
            instance.__init__(filename, frame_duration)
            return instance
