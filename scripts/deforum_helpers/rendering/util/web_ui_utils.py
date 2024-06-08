# noinspection PyUnresolvedReferences
from deforum_api import JobStatusTracker
# noinspection PyUnresolvedReferences
from modules.shared import state


WEB_UI_SLEEP_DELAY = 0.1


def init_job(init):
    state.job_count = init.args.anim_args.max_frames


def update_job(init, indexes):
    frame = indexes.frame.i + 1
    max_frames = init.args.anim_args.max_frames
    state.job = f"frame {frame}/{max_frames}"
    state.job_no = frame + 1
    if state.skipped:
        print("\n** PAUSED **")
        state.skipped = False
        while not state.skipped:
            time.sleep(WEB_UI_SLEEP_DELAY)
        print("** RESUMING **")


def update_status_tracker(init, indexes):
    progress = indexes.frame.i / init.args.anim_args.max_frames
    JobStatusTracker().update_phase(init.root.job_id, phase="GENERATING", progress=progress)


def update_progress_during_cadence(init, indexes):
    state.job = f"frame {indexes.tween.i + 1}/{init.args.anim_args.max_frames}"
    state.job_no = indexes.tween.i + 1
