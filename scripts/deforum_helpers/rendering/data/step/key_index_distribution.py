import random
from enum import Enum

from ...util import log_utils


class KeyIndexDistribution(Enum):
    UNIFORM_WITH_PARSEQ = "Uniform with Parseq"  # similar to uniform, but parseq key frame diffusion is enforced.
    UNIFORM_SPACING = "Uniform Spacing"  # distance defined by cadence
    RANDOM_SPACING = "Random Spacing"  # distance loosely based on cadence (poc)
    RANDOM_PLACEMENT = "Random Placement"  # no relation to cadence (poc)

    @property
    def name(self):
        return self.value

    @staticmethod
    def default():
        return KeyIndexDistribution.UNIFORM_WITH_PARSEQ  # same as UNIFORM_SPACING, if no Parseq keys are present

    def calculate(self, start_index, max_frames, num_key_steps, parseq_adapter):
        if self == KeyIndexDistribution.UNIFORM_WITH_PARSEQ:
            return self._uniform_with_parseq_indexes(start_index, max_frames, num_key_steps, parseq_adapter)
        if self == KeyIndexDistribution.UNIFORM_SPACING:
            return self._uniform_indexes(start_index, max_frames, num_key_steps)
        elif self == KeyIndexDistribution.RANDOM_SPACING:
            return self._random_spacing_indexes(start_index, max_frames, num_key_steps)
        elif self == KeyIndexDistribution.RANDOM_PLACEMENT:
            return self._random_placement_indexes(start_index, max_frames, num_key_steps)
        else:
            raise ValueError(f"Invalid KeyIndexDistribution: {self}")

    @staticmethod
    def _uniform_indexes(start_index, max_frames, num_key_steps):
        return [1 + start_index + int(n * (max_frames - 1 - start_index) / (num_key_steps - 1))
                for n in range(num_key_steps)]

    @staticmethod
    def _uniform_with_parseq_indexes(start_index, max_frames, num_key_steps, parseq_adapter):
        parseq_key_frames = [keyframe["frame"] for keyframe in parseq_adapter.parseq_json["keyframes"]]
        parseq_key_frames = parseq_key_frames[1:-1]  # Ignore 1st and last parseq frame, because they're keys anyway.
        num_parseq_key_frames = len(parseq_key_frames)
        num_uniform_key_frames = num_key_steps - num_parseq_key_frames
        uniform_indices = KeyIndexDistribution._uniform_indexes(start_index, max_frames, num_uniform_key_frames)
        key_frames = parseq_key_frames.copy()
        key_frames.extend(uniform_indices)  # FIXME, if this replaces a frame, we need to generated an additional one.
        key_frames.sort()
        assert len(key_frames) == num_key_steps
        return key_frames

    @staticmethod
    def _random_spacing_indexes(start_index, max_frames, num_key_steps):
        uniform_indexes = KeyIndexDistribution._uniform_indexes(start_index, max_frames, num_key_steps)
        indexes = [start_index + 1, max_frames]  # Enforce first and last indices
        total_spacing = max_frames - start_index - 1  # Calculate initial total spacing
        noise_factor = 0.5  # Higher value creates more variation
        for i in range(1, num_key_steps - 1):
            base_index = uniform_indexes[i]
            noise = random.uniform(-noise_factor, noise_factor) * (total_spacing / (num_key_steps - 1))
            index = int(base_index + noise)
            index = max(start_index + 1, min(index, max_frames - 1))
            indexes.append(index)
            total_spacing -= index - indexes[i - 1]
        indexes.sort(key=lambda key_index: key_index)
        return indexes

    @staticmethod
    def _random_placement_indexes(start_index, max_frames, num_key_steps):
        indexes = [start_index + 1, max_frames]  # Enforce first and last indices
        for _ in range(1, num_key_steps - 1):
            index = random.randint(start_index + 1, max_frames - 1)
            indexes.append(index)
        indexes.sort(key=lambda i: i)
        return indexes
