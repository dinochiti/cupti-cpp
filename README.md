# CUPTI and CUDA

The purpose of this project is to experiment with CUPTI, the Nvidia CUDA Profiling Tools Interface.

CUPTI is a dynamic C/C++ library that can be empowers CUDA developers to trace and profile their GPU code.

## Build

Clone the repo and invoke `make`.

This project is Linux-only for now. My platform is Ubuntu running on Windows Subsystem for Linux 2 (WSL2).

The full CUDA toolkit will need to be [downloaded](https://developer.nvidia.com/cuda-toolkit) and installed--it includes CUPTI (but be aware that CUPTI sometimes releases between CUDA releases, so you might like to update CUPTI to its own latest release).

## Metrics

Can any performance insights be inferred from the project so far?

Using the [`manyruns.py`](manyruns.py) Python script to invoke multiple variants (each multiple times, with results averaged after dropping the max and min times, as usual) gives the following results:

| block size  | 32       | 64       | 128      | 256      |
|-------------|----------|----------|----------|----------|
| 32 points   | 5.898    | 5.920    | 5.891    | 5.917    |
| 64  points  | 5.942    | 5.917    | 5.898    | 5.920    |
| 1K  points  | 5.981    | 5.949    | 5.949    | 6.541    |
| 1M  points  | 479.158  | 461.526  | 581.678  | 573.424  |
| 16M  points | 7545.093 | 7282.672 | 9266.500 | 9132.269 |

There's not a lot of *fine* detail here but you can see that block sizes work best when they are not too big and not too small. Further tuning for individual use cases might prove productive.

Also interesting is that execution time is very similar across all instances for sets of 32, 64 and 1024 data points. This suggests that with these smaller data sets the overhead of invoking the kernels in the first place dominates the total execution time. With the larger sets the execution time is noticable larger, as is the time variance between blak sizes.

Note that these executions were all with a step size of 1; in other words, not actively trying to cause shared memory bank conflicts.
