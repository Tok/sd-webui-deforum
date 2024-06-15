from ....generate import generate


def call_generate(init, step):
    ia = init.args
    return generate(ia.args, init.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args,
                    ia.root, init.parseq_adapter, init.indexes.frame.i, sampler_name=step.schedule.sampler_name)
