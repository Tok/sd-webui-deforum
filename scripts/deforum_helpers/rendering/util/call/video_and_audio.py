from ..utils import context
from ....video_audio_utilities import get_next_frame, render_preview


def call_render_preview(init, i, last_preview_frame):
    with context(init.args) as ia:
        return render_preview(ia.args, ia.anim_args, ia.video_args, ia.root, i, last_preview_frame)


def call_get_next_frame(init, i, video_path, is_mask: bool = False):
    return get_next_frame(init.output_directory, video_path, i, is_mask)
