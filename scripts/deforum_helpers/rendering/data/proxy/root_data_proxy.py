from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, kw_only=True)
class RootDataProxy:
    """
    An immutable data container for holding data required by "render_animation".

    This class is designed to be used with `RootDataProxyWrapper` which provides
    a mutable interface through "fake" setters.  These setters do not modify the
    original `RootDataProxy` instance, but instead create new instances with the
    updated data. This approach ensures that the original data remains immutable
    while still allowing for changes to be made.
    """

    # TODO improve typehints
    # Read-only attributes, accessed through the wrapper
    device: Any
    half_precision: Any
    timestring: Any
    mask_preset_names: Any
    frames_cache: Any
    job_id: Any
    animation_prompts: Any

    # Attributes that can be replaced by the wrapper
    noise_mask: Any
    init_sample: Any
    subseed: Any
    subseed_strength: Any
    clipseg_model: Any
    seed_internal: Any

    @staticmethod
    def create(root: Any) -> 'RootDataProxy':
        return RootDataProxy(
            device=root.device,
            half_precision=root.half_precision,
            timestring=root.timestring,
            mask_preset_names=root.mask_preset_names,
            frames_cache=root.frames_cache,
            job_id=root.job_id,
            animation_prompts=root.animation_prompts,
            noise_mask=root.noise_mask,
            init_sample=root.init_sample,
            subseed=root.subseed,
            subseed_strength=root.subseed_strength,
            clipseg_model=root.clipseg_model,
            seed_internal=root.seed_internal)


class RootDataProxyWrapper:
    """
    Provides a mutable interface to the `RootDataProxy` by replacing the
    entire proxy instance when a setter is called.
    """

    def __init__(self, root_data_proxy: RootDataProxy):
        self._proxy = root_data_proxy

    @staticmethod
    def create(root: Any) -> 'RootDataProxyWrapper':
        return RootDataProxyWrapper(RootDataProxy.create(root))

    def device(self):
        return self._proxy.device

    def half_precision(self):
        return self._proxy.half_precision

    def timestring(self):
        return self._proxy.timestring

    def mask_preset_names(self):
        return self._proxy.mask_preset_names

    def frames_cache(self):
        return self._proxy.frames_cache

    def job_id(self):
        return self._proxy.job_id

    def animation_prompts(self):
        return self._proxy.animation_prompts

    @property
    def noise_mask(self):
        return self._proxy.noise_mask

    @noise_mask.setter
    def noise_mask(self, value):
        self._proxy = replace(self._proxy, noise_mask=value)

    @property
    def init_sample(self):
        return self._proxy.init_sample

    @init_sample.setter
    def init_sample(self, value):
        self._proxy = replace(self._proxy, init_sample=value)

    @property
    def subseed(self):
        return self._proxy.subseed

    @subseed.setter
    def subseed(self, value):
        self._proxy = replace(self._proxy, subseed=value)

    @property
    def subseed_strength(self):
        return self._proxy.subseed_strength

    @subseed_strength.setter
    def subseed_strength(self, value):
        self._proxy = replace(self._proxy, subseed_strength=value)

    @property
    def clipseg_model(self):
        return self._proxy.clipseg_model

    @clipseg_model.setter
    def clipseg_model(self, value):
        self._proxy = replace(self._proxy, clipseg_model=value)

    @property
    def seed_internal(self):
        return self._proxy.seed_internal

    @seed_internal.setter
    def seed_internal(self, value):
        self._proxy = replace(self._proxy, seed_internal=value)
