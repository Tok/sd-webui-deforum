from ....generate import generate


def call_generate(init, i, schedule):
    ia = init.args
    return generate(ia.args, init.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args,
                    ia.root, init.parseq_adapter, i, sampler_name=schedule.sampler_name)
