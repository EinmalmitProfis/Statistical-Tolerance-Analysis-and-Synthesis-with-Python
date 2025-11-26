[![DOI](https://zenodo.org/badge/342926703.svg)](https://doi.org/10.5281/zenodo.17723952)


# Statistical Tolerance Analysis and Synthesis with Python

This repository contains the Python implementation accompanying

> P. Grohmann, M. S. J. Walter,  
> **“Speeding up Statistical Tolerance Analysis to Real Time”**,  
> *Applied Sciences* 11(9), 4207, 2021.  
> https://doi.org/10.3390/app11094207

The goal of this project is to provide an **open, reproducible, and extendable** implementation that can be used to improve existing tolerance analysis and synthesis workflows **without relying on proprietary software**.

---

## Key Features

- 🔍 **Statistical tolerance analysis**  
  Evaluate the resulting closing dimension of an assembly based on stochastic tolerances.

- 🧮 **Statistical tolerance synthesis (optimization)**  
  Find cost-efficient tolerance vectors that satisfy a statistical constraint (upper specification limit for the standard deviation).

- 🧠 **Unified core model**  
  Shared core logic in `tolerance_optimization/common_analysis.py` for both CPU (NumPy) and GPU (CuPy) backends.

- ⚡ **CPU and GPU backends**
  - **NumPy** for fast CPU-based simulations.
  - **CuPy** for high-performance GPU-based simulations (NVIDIA CUDA).

- 🔁 **Differential Evolution optimizer**  
  Uses `scipy.optimize.differential_evolution` to perform global tolerance synthesis.

- 🧩 **Easily adaptable to other tolerance problems**  
  By modifying the closing dimension equation and the cost function, the same framework can be applied to different assemblies.

---

## How It Works

To achieve high-performance simulations, this project uses:

- **NumPy** for the CPU implementation (`numpy_analysis.py`)
- **CuPy** for the GPU implementation (`cupy_analysis.py`)
- **SciPy**’s **Differential Evolution** algorithm for tolerance optimization (`scipy.optimize.differential_evolution`)

At the core, the class `CommonAnalysis` in  
`tolerance_optimization/common_analysis.py` defines:

- The **probability distributions** of individual dimensions
- The **closing dimension** (assembly function) via `closing_dimension(...)`
- The **cost model** via `costs(...)`, `cost_function(...)`, and `cost_function_tolerance_optimization(...)`
- A **constraint** on the standard deviation of the closing dimension (`usl` – upper specification limit)

### Distributions

The vector `distributions` encodes which distribution is used per dimension:

- `0` → uniform distribution  
- `1` → normal distribution  

These are implemented in:

- `tolerance_optimization/numpy_analysis.py`  
- `tolerance_optimization/cupy_analysis.py`  

### Closing Dimension

The method

```python
def closing_dimension(self, tolerances):
    ...
    return self.backend.minimum(a1, a2)
````

defines the assembly’s closing dimension as a function of the sampled tolerances.
This is where you would adapt the code if your mechanical/geometrical model differs from the one in the publication.

---

## Repository Structure

```text
.
├── stat_tol_analysis_numpy.py      # Statistical analysis on CPU (NumPy)
├── stat_tol_analysis_cupy.py       # Statistical analysis on GPU (CuPy)
├── stat_tol_synthesis_numpy.py     # Statistical tolerance synthesis on CPU
├── stat_tol_synthesis_cupy.py      # Statistical tolerance synthesis on GPU
├── project_export.py               # Helper: create a Markdown project overview
└── tolerance_optimization/
    ├── __init__.py
    ├── common_analysis.py          # Core logic (model, cost, constraints)
    ├── numpy_analysis.py           # NumPy backend
    └── cupy_analysis.py            # CuPy backend
```

---

## Example Usage

### 1. Statistical Tolerance Analysis (no optimization)

These scripts perform repeated simulations for increasing sample sizes and print the current sample size to the console. You can use them to benchmark performance or to collect statistics.

#### CPU (NumPy)

```bash
python stat_tol_analysis_numpy.py
```

This will:

* Instantiate `StatisticalToleranceSynthesisNumPy`
* Generate random samples for each dimension
* Compute the closing dimension
* Repeat this for various `sample_size` values

#### GPU (CuPy)

```bash
python stat_tol_analysis_cupy.py
```

This version:

* Uses the same model and parameters, but with **CuPy** on the GPU
* Frees GPU memory between runs (`mempool.free_all_blocks()`)
* Synchronizes the device (`dev.synchronize()`) to ensure all GPU work is completed before timing/printing

---

### 2. Statistical Tolerance Synthesis (optimization)

These scripts run the **Differential Evolution** algorithm to find an optimal tolerance vector that:

* Minimizes the total cost
* Satisfies the statistical constraint (`usl >= std(closing_dimension)`)

Run:

#### CPU (NumPy)

```bash
python stat_tol_synthesis_numpy.py
```

#### GPU (CuPy)

```bash
python stat_tol_synthesis_cupy.py
```

---

## Using the Core Library in Your Own Code

You can re-use the `StatisticalToleranceSynthesis` classes directly in your own scripts.

### NumPy example

```python
import numpy as np
from tolerance_optimization.numpy_analysis import StatisticalToleranceSynthesis

mean = [7.5, 5.1, 17.5, 5.1, 5.05, 12.5, 5.1]
variable_costs = [1, 9, 5, 15, 2, 11, 18]
distributions = [1, 0, 1, 0, 1, 1, 0]  # 1: normal, 0: uniform
usl = 0.1
sample_size = 1_000_000
float_type = np.float32

analysis = StatisticalToleranceSynthesis(
    mean=mean,
    distributions=distributions,
    sample_size=sample_size,
    variable_costs=variable_costs,
    fixed_costs=0.0,
    usl=usl,
    float_type=float_type,
)

opti_vector = [0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]

# Evaluate cost for a given tolerance vector
cost = analysis.cost_function_tolerance_optimization(opti_vector)
print("Cost:", cost)
```

The same pattern applies to the CuPy backend, replacing NumPy with CuPy:

```python
import cupy as cp
from tolerance_optimization.cupy_analysis import StatisticalToleranceSynthesis

analysis = StatisticalToleranceSynthesis(
    mean=mean,
    distributions=distributions,
    sample_size=sample_size,
    variable_costs=variable_costs,
    fixed_costs=0.0,
    usl=usl,
    float_type=cp.float32,
)
```

---

## Adapting the Code to Other Tolerance Problems

The current implementation solves the tolerance problem described in the referenced publication.
To adapt the model to your own assembly or tolerance problem, you will mainly modify:

1. **Closing dimension**
   In `tolerance_optimization/common_analysis.py`, change:

   ```python
   def closing_dimension(self, tolerances):
       ...
       return self.backend.minimum(a1, a2)
   ```

   to reflect your own functional dimension / stack-up equation.

2. **Cost model**

   * `costs(self, opti_vector)` – cost per tolerance
   * `cost_function_without(self, opti_vector)` – total cost without constraints
   * `cost_function_tolerance_optimization(self, opti_vector)` – cost with statistical constraint

3. **Default parameters**
   Optionally adjust `make_analysis(...)` in `CommonAnalysis` to define new default parameters for `mean`, `variable_costs`, `distributions`, etc.

---

## 📜 License & Attribution

This repository is licensed under the **MIT License**.

Please also cite the corresponding paper if you use this code for research.

---

## Related Publication

The methods, case study (overconstrained door hinge assembly), and runtime studies implemented in this repository are described in detail in:

* **Grohmann, P.; Walter, M. S. J.**
  *Speeding up Statistical Tolerance Analysis to Real Time.*
  *Applied Sciences* 2021, 11(9), 4207.
  [https://doi.org/10.3390/app11094207](https://doi.org/10.3390/app11094207)

The article is **open access** under the **Creative Commons Attribution (CC BY 4.0)** license.

---

## Origin of this Work

The methodology and case study implemented in this repository build on the research of **Michael S. J. Walter** on statistical tolerance analysis and synthesis.
The Python implementation provided here grew out of the joint work by **Peter Grohmann** and **Michael S. J. Walter** that is documented in the publication cited above.

The code base in this repository was then developed further in Python by **Peter Grohmann** together with **Rolands Kalvāns**, who co-designed and implemented the NumPy/CuPy backends and the associated optimization workflows.

---

### Suggested citation

Plain text:

> Grohmann, P.; Walter, M. S. J. (2021). Speeding up Statistical Tolerance Analysis to Real Time. *Applied Sciences*, 11(9), 4207. [https://doi.org/10.3390/app11094207](https://doi.org/10.3390/app11094207)

BibTeX:

```bibtex
@article{GrohmannWalter2021,
  author  = {Grohmann, Peter and Walter, Michael S. J.},
  title   = {Speeding up Statistical Tolerance Analysis to Real Time},
  journal = {Applied Sciences},
  year    = {2021},
  volume  = {11},
  number  = {9},
  pages   = {4207},
  doi     = {10.3390/app11094207}
}
```
