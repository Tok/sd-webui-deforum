import random
from enum import Enum


class KeyIndexDistribution(Enum):
    UNIFORM_SPACING = "Uniform Spacing"
    RANDOM_SPACING = "Random Spacing"
    RANDOM_PLACEMENT = "Random Placement"

    @property
    def name(self):
        return self.value

    @staticmethod
    def default():
        return KeyIndexDistribution.UNIFORM_SPACING

    def calculate(self, start_index, max_frames, num_key_steps):
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
