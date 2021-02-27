import cupy as cp
from .common_analysis import CommonAnalysis


class StatisticalToleranceSynthesis(CommonAnalysis):
    backend = cp
    float_type = cp.float32

    def uniform_distribution(self, mean, opti_vector):
        low = mean - opti_vector / 2
        high = mean + opti_vector / 2
        return cp.random.uniform(
            low=low, high=high, size=self.sample_size, dtype=self.float_type
        )

    def normal_distribution(self, mean, opti_vector):
        scale = opti_vector / 6
        return cp.random.normal(
            loc=mean, scale=scale, size=self.sample_size, dtype=self.float_type
        )
