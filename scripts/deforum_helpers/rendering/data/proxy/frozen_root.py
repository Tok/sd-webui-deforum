import modules.paths as ph
import modules.shared as sh
import os
import tempfile

from scripts.deforum_helpers.general_utils import get_os


class FrozenDict(dict):
    """An immutable dictionary that raises an error when trying to modify it."""
    def __setitem__(self, key, value):
        raise TypeError("FrozenDict is immutable")

    def __delitem__(self, key):
        raise TypeError("FrozenDict is immutable")

    def pop(self, *args):
        raise TypeError("FrozenDict is immutable")

    def popitem(self):
        raise TypeError("FrozenDict is immutable")

    def clear(self):
        raise TypeError("FrozenDict is immutable")

    def update(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable")


def RootArgs():
    return FrozenDict({
        "device": sh.device,
        "models_path": ph.models_path + '/Deforum',
        "half_precision": not sh.cmd_opts.no_half,
        "clipseg_model": None,
        "mask_preset_names": ('everywhere', 'video_mask'),
        "frames_cache": [],
        "raw_batch_name": None,
        "raw_seed": None,
        "timestring": "",
        "subseed": -1,
        "subseed_strength": 0,
        "seed_internal": 0,
        "init_sample": None,
        "noise_mask": None,
        "initial_info": None,
        "first_frame": None,
        "animation_prompts": None,
        "current_user_os": get_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    })