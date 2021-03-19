import cupy as cp
from tolerance_optimization.cupy_analysis import StatisticalToleranceSynthesis as StatisticalToleranceSynthesisCuPy

solver = StatisticalToleranceSynthesisCuPy
dev = cp.cuda.Device()
floattype = cp.float32


def main():
    mempool = cp.get_default_memory_pool()
    mempool.get_limit()

    opti_vector = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    mean = [7.5, 5.1, 17.5, 5.1, 5.05, 12.5, 5.1]
    variable_costs = [1, 9, 5, 15, 2, 11, 18]
    distributions = [1, 0, 1, 0, 1, 1, 0]
    repeat = 25

    for k in range(1):
        i = 1
        sample_size = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 15000000, 20000000, 25000000,
                       30000000, 35000000, 40000000, 45000000, 50000000, 55000000, 60000000, 65000000, 70000000,
                       75000000, 80000000, 85000000, 90000000]
        for index, samplesize in enumerate(sample_size):
            for l in range(repeat):
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
                c.closing_dimension(c.tolerances(opti_vector=opti_vector))
                dev.synchronize()
                print(samplesize)


if __name__ == "__main__":
    main()
