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
        """Calculates uniform indices according to cadence, but parseq key frames replace the closest deforum key."""
        uniform_indices = KeyIndexDistribution._uniform_indexes(start_index, max_frames, num_key_steps)
        if not parseq_adapter.use_parseq:
            log_utils.warn("UNIFORM_WITH_PARSEQ, but Parseq is not active, using UNIFORM_SPACING instead.")
            return uniform_indices

        parseq_key_frames = [keyframe["frame"] for keyframe in parseq_adapter.parseq_json["keyframes"]]
        shifted_parseq_frames = [frame + 1 for frame in parseq_key_frames]
        key_frames_set = set(uniform_indices)  # set for faster membership checks

        # Insert parseq keyframes while maintaining keyframe count
        for current_frame in shifted_parseq_frames:
            if current_frame not in key_frames_set:
                # Find the closest index in the set to replace
                closest_index = min(key_frames_set, key=lambda x: abs(x - current_frame))
                key_frames_set.remove(closest_index)
                key_frames_set.add(current_frame)

        key_frames = list(key_frames_set)
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
