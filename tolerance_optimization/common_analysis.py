from itertools import starmap
from typing import List, NamedTuple
import numpy as np

ModelResults = NamedTuple(
    "ModelResults",
    (
        ("costs", np.ndarray),
        ("cost", float),
        ("sm", float),
        ("opti_vector", np.ndarray),
    ),
)
OptiVector = NamedTuple("OptiVector", (("opti", np.ndarray), ("model", ModelResults)))


class CommonAnalysis:
    variable_costs: List[int]
    fixed_costs: float
    usl: float
    distributions: List[int]
    mean: List[int]
    backend: np

    float_type = None
    _tolerances = None

    def __init__(
            self,
            mean: List[float],
            distributions: List[int],
            sample_size: int,
            variable_costs: List[int],
            fixed_costs: float,
            usl: float,
            float_type=None,
    ):
        if float_type is None:
            float_type = self.float_type
        (
            self.mean,
            self.distributions,
            self.sample_size,
            self.variable_costs,
            self.fixed_costs,
            self.usl,
            self.float_type,
        ) = (
            mean,
            distributions,
            sample_size,
            variable_costs,
            fixed_costs,
            usl,
            float_type,
        )
        self.initialize_tolerances()

    def initialize_tolerances(self) -> None:
        self._tolerances = self.backend.empty(
            shape=(self.sample_size, len(self.distributions)),
            order="F",      #change the storage layout here
            dtype=self.float_type,
        )

    @classmethod
    def make_analysis(cls, sample_size):
        mean = [7.5, 5.1, 17.5, 5.1, 5.05, 12.5, 5.1]
        variable_costs = [1, 9, 5, 15, 2, 11, 18]
        distributions = [1, 0, 1, 0, 1, 1, 0]

        return cls(
            sample_size=sample_size,
            mean=mean,
            fixed_costs=0,
            variable_costs=variable_costs,
            distributions=distributions,
            usl=0.1,
        )

    def uniform_distribution(self, mean, opti_vector):
        raise NotImplementedError

    def normal_distribution(self, mean, opti_vector):
        raise NotImplementedError

    def constraints(self, opti_vector):
        return self.backend.std(
            self.closing_dimension(self.tolerances(opti_vector=opti_vector))
        )

    def fill_array(self, distribution, mean, opti_vector):
        return (self.uniform_distribution, self.normal_distribution)[distribution](
            mean=mean, opti_vector=opti_vector
        )

    def tolerances(self, opti_vector):

        for index, data in enumerate(
                starmap(self.fill_array, zip(self.distributions, self.mean, opti_vector))
        ):
            self._tolerances[:, index] = data
        return self._tolerances

    def costs(self, opti_vector):
        return np.array(self.variable_costs) / np.array(opti_vector)

    def cost_function_without(self, opti_vector):
        return self.costs(opti_vector).sum() + self.fixed_costs

    def closing_dimension(self, tolerances):
        x0 = tolerances[:, 0]
        x1 = tolerances[:, 1]
        x2 = tolerances[:, 2]
        x3 = tolerances[:, 3]
        x4 = tolerances[:, 4]
        x5 = tolerances[:, 5]
        x6 = tolerances[:, 6]

        a1 = (x5 + 0.5 * x6) - (x2 + 0.5 * x3)
        a2 = x4 - (x0 + 0.5 * x1)

        return self.backend.minimum(a1, a2)

    def cost_function(self, opti_vector) -> float:
        if any(np.isnan(opti_element) for opti_element in opti_vector):
            return np.infty
        elif self.usl >= self.constraints(opti_vector):
            return self.cost_function_without(opti_vector)
        return np.infty

    def cost_function_tolerance_optimization(self, opti_vector) -> float:
        if self.usl >= self.constraints(opti_vector):
            return self.costs(opti_vector).sum() + self.fixed_costs
        return np.infty
