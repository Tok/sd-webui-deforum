from ....resume import get_resume_vars


def call_get_resume_vars(init, turbo):
    return get_resume_vars(folder=init.args.args.outdir,
                           timestring=init.args.anim_args.resume_timestring,
                           cadence=turbo.steps)
