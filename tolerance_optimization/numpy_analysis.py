import numpy as np

from .common_analysis import CommonAnalysis


class StatisticalToleranceSynthesis(CommonAnalysis):
    backend = np
    float_type = np.float32

    def uniform_distribution(self, mean, opti_vector):
        low = mean - opti_vector / 2
        high = mean + opti_vector / 2
        return np.random.uniform(low=low, high=high, size=self.sample_size)

    def normal_distribution(self, mean, opti_vector):
        scale = opti_vector / 6
        return np.random.normal(loc=mean, scale=scale, size=self.sample_size)
