import dataclasses

from typing import Optional, Any


@dataclasses.dataclass(init=True, frozen=True, repr=False, eq=False)
class Schedule:
    steps: int
    sampler_name: str
    clipskip: int
    noise_multiplier: float
    eta_ddim: float
    eta_ancestral: float  # TODO unify ddim- and a-eta to use one or the other, depending on sampler
    mask: Optional[Any]
    noise_mask: Optional[Any]

    def __new__(cls, *args, **kwargs):  # locks the normal constructor to enforce proper initialization
        raise TypeError("Use Schedule.create() to create new instances.")

    @classmethod
    def create(cls, keys, i, anim_args, args):
        # TODO typecheck keys as DeformAnimKeys or provide key collection or something
        """Create a new Schedule instance based on the provided parameters."""
        steps = cls.schedule_steps(keys, i, anim_args)
        sampler_name = cls.schedule_sampler(keys, i, anim_args)
        clipskip = cls.schedule_clipskip(keys, i, anim_args)
        noise_multiplier = cls.schedule_noise_multiplier(keys, i, anim_args)
        eta_ddim = cls.schedule_ddim_eta(keys, i, anim_args)
        eta_ancestral = cls.schedule_ancestral_eta(keys, i, anim_args)
        mask = cls.schedule_mask(keys, i, args)  # TODO for some reason use_mask is in args instead of anim_args
        noise_mask = cls.schedule_noise_mask(keys, i, anim_args)

        instance = object.__new__(cls)  # creating the instance without raising the type error defined in __new__.
        instance.__init__(steps, sampler_name, clipskip, noise_multiplier, eta_ddim, eta_ancestral, mask, noise_mask)
        return instance

    @classmethod
    def _has_schedule(cls, keys, i):
        return keys.steps_schedule_series[i] is not None

    @classmethod
    def _has_mask_schedule(cls, keys, i):
        return keys.mask_schedule_series[i] is not None

    @classmethod
    def _has_noise_mask_schedule(cls, keys, i):
        return keys.noise_mask_schedule_series[i] is not None

    @classmethod
    def _use_on_cond_if_scheduled(cls, keys, i, value, cond):
        return value if cond and cls._has_schedule(keys, i) else None

    @classmethod
    def schedule_steps(cls, keys, i, anim_args):
        return cls._use_on_cond_if_scheduled(keys, i, int(keys.steps_schedule_series[i]),
                                    anim_args.enable_steps_scheduling)

    @classmethod
    def schedule_sampler(cls, keys, i, anim_args):
        return cls._use_on_cond_if_scheduled(keys, i, keys.sampler_schedule_series[i].casefold(),
                                    anim_args.enable_sampler_scheduling)

    @classmethod
    def schedule_clipskip(cls, keys, i, anim_args):
        return cls._use_on_cond_if_scheduled(keys, i, int(keys.clipskip_schedule_series[i]),
                                    anim_args.enable_clipskip_scheduling)

    @classmethod
    def schedule_noise_multiplier(cls, keys, i, anim_args):
        return cls._use_on_cond_if_scheduled(keys, i, float(keys.noise_multiplier_schedule_series[i]),
                                    anim_args.enable_noise_multiplier_scheduling)

    @classmethod
    def schedule_ddim_eta(cls, keys, i, anim_args):
        return cls._use_on_cond_if_scheduled(keys, i, float(keys.ddim_eta_schedule_series[i]),
                                    anim_args.enable_ddim_eta_scheduling)

    @classmethod
    def schedule_ancestral_eta(cls, keys, i, anim_args):
        return cls._use_on_cond_if_scheduled(keys, i, float(keys.ancestral_eta_schedule_series[i]),
                                    anim_args.enable_ancestral_eta_scheduling)

    @classmethod
    def schedule_mask(cls, keys, i, args):
        # TODO can we have a mask schedule without a normal schedule? if so check and optimize
        return keys.mask_schedule_series[i] \
            if args.use_mask and cls._has_mask_schedule(keys, i) else None

    @classmethod
    def schedule_noise_mask(cls, keys, i, anim_args):
        #TODO can we have a noise mask schedule without a mask- and normal schedule? if so check and optimize
        return keys.noise_mask_schedule_series[i] \
            if anim_args.use_noise_mask and cls._has_noise_mask_schedule(keys, i) else None
