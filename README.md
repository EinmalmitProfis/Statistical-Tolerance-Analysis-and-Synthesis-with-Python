# Statistical-Tolerance-Analysis-and-Synthesis-with-Python

This repository contains the Python code for Statistical Tolerance Analysis and Statistical Tolerance Synthesis discussed in the MDPI publication [Speeding up Statistical Tolerance Analysis to Real Time](https://www.mdpi.com/2076-3417/11/9/4207). It can be used to improve existing Statistical Tolerance Analysis and Statistical Tolerance Synthesis workflows without having to resort to proprietary software solutions. 

## How it works
In order to achieve high-performance computing with Python, the [NumPy](https://numpy.org/) library is used for the CPU implementation. We also use [CuPy](https://cupy.dev/) for a convenient implementation on the GPU. For the tolerance optimization procedure, we rely on the Differential evolution algorithm from [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html). 

This presented code provides a solution to the tolerance problem described in the publication linked above. It can of course be adapted to other tolerance problems by modifying the closed component equation and the cost function in common_analysis.py.

