# Assessment 2: Operator Optimization and Methods

## FlagGems Operator Development Competition — Track 1

### 1. Introduction and Objectives
For Assessment 2, our primary goals were to address performance bottlenecks identified in Assessment 1 and expand our operator coverage. Specifically, we focused on:
1. **Optimizing the `median` operator**: The baseline implementation achieved only a ~0.3x speedup compared to PyTorch because it relied on full sorting (Bitonic and Radix Sort). We transitioned to a highly optimized selection algorithm to surpass the required $\ge 0.9x$ speedup threshold.
2. **Implementing the `log10` operator**: Developing a new, highly optimized pointwise operator that seamlessly handles type promotion, N-dimensional tiling, and broadcasting.

### 2. Techniques Tried

#### A. Median Optimization: From Sorting to Selection
- **Full Sorting (Baseline):** Initially used Bitonic and Radix Sort ($O(N \log N)$). Failed to achieve speedup due to the overhead of sorting every element when only rank $K = \lfloor(N-1)/2\rfloor$ is needed.
- **QuickSelect (Prototype):** Attempted an $O(N)$ QuickSelect. Abandoned due to high thread divergence and uncoalesced memory accesses on the GPU SIMT architecture.
- **Radix Select in Registers (Final `medianV2`):** Developed a strictly register-bound $O(N)$ Radix Select. It loops over the 32 bits of a monotonic integer representation of the floats, pruning active lanes based on pop-counts. This completely eliminated HBM round-trips for intermediate bins and achieved speedups of ~1.1x - 1.5x over PyTorch.
- **Edge Case Canonicalization (V3):** During extensive unit testing with duplicates, we discovered discrepancies between PyTorch's native high-level float logic and our bit-level sorting. Specifically, floats like `-0.0` and `0.0` evaluate equally in Python, but their integer bit representations differ. Additionally, `NaN` values carry varying bit payloads. We patched the kernel by standardizing `-0.0` to `0.0` and mapping all NaNs to a canonical positive NaN pattern *before* bitcasting to `uint32`. This ensured 100% stable matching with PyTorch's native subset evaluation.

#### B. Log10 Implementation: Dynamic Pointwise Generation
- **Pointwise Dynamic Wrapper:** Instead of writing a manual kernel for every tensor shape, we leveraged FlagGems' `pointwise_dynamic` decorator. This automatically generates optimal grid/block configurations for contiguous, strided, and non-contiguous memory layouts.
- **Constant Multiplication:** Rather than calling a complex base-10 logarithm instruction, we computed `log10(x) = ln(x) * log10(e)`. Multiplying by the constant `0.4342944819032518` is significantly faster at the register level.
- **Type Promotion:** Integrated `INT_TO_FLOAT` promotion directly into the wrapper to ensure integer tensors are automatically cast to `tl.float32` before the `tl.log` math function is applied, matching PyTorch's native behavior exactly.

### 3. Insights Gained

1. **Pass Efficiency over Algorithmic Asymptotics:** For GPU kernels (like our `medianV2`), minimizing the number of passes over data in registers drastically reduces instruction counts and latency, even if the theoretical complexity involves a higher constant factor (32 bit passes).
2. **Framework Abstraction:** Utilizing Triton's JIT combined with FlagGems' decorators drastically reduces boilerplate. For `log10`, we achieved maximum memory bandwidth utilization without writing explicit boundary checks or tiling loops.
3. **Mathematical Simplifications:** Converting base-10 logarithms to natural logarithms with a constant multiplier proved to be highly efficient for ALUs.

### 4. Implementation Details (Updated GitHub Repo)

The proposed methods have been implemented in the `flag_gems` repository:
- **Median V2:** `src/flag_gems/ops/medianV2.py`
- **Log10:** Implementation developed and validated in `flagos-log10-triton-kernel.ipynb`, ready for integration into `src/flag_gems/ops/log10.py`.

The iterative development log, containing exact hyperparameter settings and experimental results, can be found in `assessment2_dev_log.md`.

*Note: Please refer to the updated `assessment2_dev_log.md` for specific benchmark scores and the latest Kaggle Leaderboard screenshot.*
