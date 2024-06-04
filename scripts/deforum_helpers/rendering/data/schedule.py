from dataclasses import dataclass
from typing import Optional, Any


@dataclass(init=True, frozen=True, repr=False, eq=False)
class Schedule:
    steps: int
    sampler_name: str
    clipskip: int
    noise_multiplier: float
    eta_ddim: float
    eta_ancestral: float  # TODO unify ddim- and a-eta to use one or the other, depending on sampler
    mask: Optional[Any]
    noise_mask: Optional[Any]

    @staticmethod
    def create(keys, i, anim_args, args):
        # TODO typecheck keys as DeformAnimKeys or provide key collection or something
        """Create a new Schedule instance based on the provided parameters."""
        steps = Schedule.schedule_steps(keys, i, anim_args)
        sampler_name = Schedule.schedule_sampler(keys, i, anim_args)
        clipskip = Schedule.schedule_clipskip(keys, i, anim_args)
        noise_multiplier = Schedule.schedule_noise_multiplier(keys, i, anim_args)
        eta_ddim = Schedule.schedule_ddim_eta(keys, i, anim_args)
        eta_ancestral = Schedule.schedule_ancestral_eta(keys, i, anim_args)
        mask = Schedule.schedule_mask(keys, i, args)  # TODO for some reason use_mask is in args instead of anim_args
        noise_mask = Schedule.schedule_noise_mask(keys, i, anim_args)
        return Schedule(steps, sampler_name, clipskip, noise_multiplier, eta_ddim, eta_ancestral, mask, noise_mask)

    @staticmethod
    def _has_schedule(keys, i):
        return keys.steps_schedule_series[i] is not None

    @staticmethod
    def _has_mask_schedule(keys, i):
        return keys.mask_schedule_series[i] is not None

    @staticmethod
    def _has_noise_mask_schedule(keys, i):
        return keys.noise_mask_schedule_series[i] is not None

    @staticmethod
    def _use_on_cond_if_scheduled(keys, i, value, cond):
        return value if cond and Schedule._has_schedule(keys, i) else None

    @staticmethod
    def schedule_steps(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, int(keys.steps_schedule_series[i]),
                                                  anim_args.enable_steps_scheduling)

    @staticmethod
    def schedule_sampler(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, keys.sampler_schedule_series[i].casefold(),
                                                  anim_args.enable_sampler_scheduling)

    @staticmethod
    def schedule_clipskip(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, int(keys.clipskip_schedule_series[i]),
                                                  anim_args.enable_clipskip_scheduling)

    @staticmethod
    def schedule_noise_multiplier(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, float(keys.noise_multiplier_schedule_series[i]),
                                                  anim_args.enable_noise_multiplier_scheduling)

    @staticmethod
    def schedule_ddim_eta(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, float(keys.ddim_eta_schedule_series[i]),
                                                  anim_args.enable_ddim_eta_scheduling)

    @staticmethod
    def schedule_ancestral_eta(keys, i, anim_args):
        return Schedule._use_on_cond_if_scheduled(keys, i, float(keys.ancestral_eta_schedule_series[i]),
                                                  anim_args.enable_ancestral_eta_scheduling)

    @staticmethod
    def schedule_mask(keys, i, args):
        # TODO can we have a mask schedule without a normal schedule? if so check and optimize
        return keys.mask_schedule_series[i] \
            if args.use_mask and Schedule._has_mask_schedule(keys, i) else None

    @staticmethod
    def schedule_noise_mask(keys, i, anim_args):
        # TODO can we have a noise mask schedule without a mask- and normal schedule? if so check and optimize
        return keys.noise_mask_schedule_series[i] \
            if anim_args.use_noise_mask and Schedule._has_noise_mask_schedule(keys, i) else None
