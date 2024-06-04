from dataclasses import dataclass
from ....animation_key_frames import DeformAnimKeys, LooperAnimKeys


@dataclass(init=True, frozen=True, repr=False, eq=False)
class AnimationKeys:
    deform_keys: DeformAnimKeys
    looper_keys: LooperAnimKeys

    def update(self, i: int):
        self.looper_keys.use_looper = self.looper_keys.use_looper
        self.looper_keys.imagesToKeyframe = self.looper_keys.imagesToKeyframe
        # TODO FIXME refactor index handling and remove new boilerplate
        self.looper_keys.imageStrength = self.looper_keys.image_strength_schedule_series[i]
        self.looper_keys.blendFactorMax = self.looper_keys.blendFactorMax_series[i]
        self.looper_keys.blendFactorSlope = self.looper_keys.blendFactorSlope_series[i]
        self.looper_keys.tweeningFramesSchedule = self.looper_keys.tweening_frames_schedule_series[i]
        self.looper_keys.colorCorrectionFactor = self.looper_keys.color_correction_factor_series[i]

    @staticmethod
    def choose_default_or_parseq_keys(default_keys, parseq_keys, parseq_adapter):
        return default_keys if not parseq_adapter.use_parseq else parseq_keys

    @staticmethod
    def from_args(step_args, parseq_adapter, seed):
        # Parseq keys are decorated, see ParseqAinmKeysDecorator and ParseqLooperKeysDecorator
        return AnimationKeys(
            AnimationKeys.choose_default_or_parseq_keys(DeformAnimKeys(step_args.anim_args, seed),
                                                        parseq_adapter.anim_keys, parseq_adapter),
            AnimationKeys.choose_default_or_parseq_keys(LooperAnimKeys(step_args.loop_args, step_args.anim_args, seed),
                                                        parseq_adapter.looper_keys, parseq_adapter))
