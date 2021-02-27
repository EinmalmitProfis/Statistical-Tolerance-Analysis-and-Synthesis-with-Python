import scipy.optimize
import cupy as cp
from tolerance_optimization.cupy_analysis import StatisticalToleranceSynthesis as StatisticalToleranceSynthesisCuPy

solver = StatisticalToleranceSynthesisCuPy

dev = cp.cuda.Device()


def callback(*args, **kwargs):
    dev.synchronize()
    print(args, kwargs)


floattype = cp.float32


def main():
    mempool = cp.get_default_memory_pool()
    mempool.get_limit()

    opti_vector = [0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]
    mean = [7.5, 5.1, 17.5, 5.1, 5.05, 12.5, 5.1]
    variable_costs = [1, 9, 5, 15, 2, 11, 18]
    distributions = [1, 0, 1, 0, 1, 1, 0]
    bounds = [(0, 0.7)] * len(opti_vector)

    sample_size = [
        10_000,
        50_000,
        100_000,
        500_000,
        1_000_000,
        2_000_000,
        4_000_000,
        6_000_000,
        8_000_000,
        10_000_000,
    ]

    number_of_runs = 10
    for samplesize in sample_size:
        for k in range(number_of_runs):
            mempool.free_all_blocks()
            c = solver(
                sample_size=samplesize,
                mean=mean,
                fixed_costs=0,
                variable_costs=variable_costs,
                distributions=distributions,
                usl=0.1,
                float_type=floattype,
            )
            result = scipy.optimize.differential_evolution(
                c.cost_function_tolerance_optimization,
                bounds,
                strategy="best1bin",
                maxiter=100,
                popsize=25,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.5,
                seed=1992,
                callback=callback,
                disp=True,
                polish=False,
                init="latinhypercube",
                updating="deferred",
                atol=0,
                workers=1,
            )


if __name__ == "__main__":
    main()
