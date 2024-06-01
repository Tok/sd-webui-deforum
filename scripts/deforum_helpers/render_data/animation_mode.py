import dataclasses


@dataclasses.dataclass
class AnimationMode:
    filename: str
    prev_flow: str
