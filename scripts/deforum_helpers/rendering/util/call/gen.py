from ....generate import generate


def call_generate(data, step):
    ia = data.args
    return generate(ia.args, data.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args,
                    ia.root, data.parseq_adapter, data.indexes.frame.i, sampler_name=step.schedule.sampler_name)
