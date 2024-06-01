from typing import Optional, Any


class Schedule:
    def __init__(self, steps: int, sampler_name: str, clipskip: int,
                 noise_multiplier: float, eta_ddim: float, eta_ancestral: float,
                 mask: Optional[Any] = None, noise_mask: Optional[Any] = None):
        self._steps = steps
        self._sampler_name = sampler_name
        self._clipskip = clipskip
        self._noise_multiplier = noise_multiplier
        self._eta_ddim = eta_ddim
        self._eta_ancestral = eta_ancestral  # TODO unify ddim- and a-eta to use one or the other, depending on sampler
        self._mask = mask
        self._noise_mask = noise_mask

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def sampler_name(self) -> str:
        return self._sampler_name

    @property
    def clipskip(self) -> int:
        return self._clipskip

    @property
    def noise_multiplier(self) -> float:
        return self._noise_multiplier

    @property
    def eta_ddim(self) -> float:
        return self._eta_ddim

    @property
    def eta_ancestral(self) -> float:
        return self._eta_ancestral

    @property
    def mask(self) -> Optional[Any]:
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def noise_mask(self) -> Optional[Any]:
        return self._noise_mask

    @noise_mask.setter
    def noise_mask(self, value):
        self._noise_mask = value
