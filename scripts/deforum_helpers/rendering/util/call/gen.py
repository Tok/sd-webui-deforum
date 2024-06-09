from ..utils import context
from ....generate import generate


def call_generate(init, i, schedule):
    with context(init.args) as ia:
        return generate(ia.args, init.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args,
                        ia.root, init.parseq_adapter, i, sampler_name=schedule.sampler_name)
