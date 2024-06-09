from enum import Enum
from pathlib import Path

from ..data import Indexes
from ..data.step import StepInit
from ...video_audio_utilities import get_frame_name


class FileFormat(Enum):
    JPG = "jpg"
    PNG = "png"

    @staticmethod
    def frame_format():
        return FileFormat.PNG

    @staticmethod
    def video_frame_format():
        return FileFormat.JPG


def _frame_filename_index(i: int, file_format: FileFormat) -> str:
    return f"{i:09}.{file_format.value}"


def _frame_filename(init: StepInit, i: int, is_depth=False, file_format=FileFormat.frame_format()) -> str:
    infix = "_depth_" if is_depth else "_"
    return f"{init.root.timestring}{infix}{_frame_filename_index(i, file_format)}"


def frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.frame.i)


def depth_frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.frame.i, True)


def tween_frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.tween.i)


def tween_depth_frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.tween.i, True)


def preview_video_image_path(init: StepInit, indexes: Indexes) -> Path:
    frame_name = get_frame_name(init.args.anim_args.video_init_path)
    index = _frame_filename_index(indexes.frame.i, FileFormat.video_frame_format())
    return Path(init.output_directory) / "inputframes" / (frame_name + index)
