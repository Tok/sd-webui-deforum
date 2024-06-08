from enum import Enum
from ..initialization import StepInit
from ..data import Indexes
from ...video_audio_utilities import get_frame_name
from pathlib import Path


class FileFormat(Enum):
    JPG = "jpg"
    PNG = "png"


def _frame_filename_index(i: int, file_format: FileFormat) -> str:
    return f"{i:09}.{file_format.value}"


def _frame_filename(init: StepInit, i: int, is_depth: bool = False, file_format: FileFormat = FileFormat.PNG) -> str:
    infix = "_depth_" if is_depth else "_"
    return f"{init.root.timestring}{infix}{_frame_filename_index(i, file_format)}"


def frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.frame.i)


def depth_frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.frame.i, True)


def tween_frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.tween.i)


def tween_depth_frame(init: StepInit, indexes: Indexes) -> str:
    return _frame_filename(init, indexes.tween.i,True)


def preview_video_image_path(init: StepInit, indexes: Indexes) -> Path:
    frame_name = get_frame_name(init.args.anim_args.video_init_path)
    return Path(init.output_directory) / "inputframes" / (frame_name + _frame_filename_index(indexes, FileFormat.JPG))
