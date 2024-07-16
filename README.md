# CUPTI and CUDA

The purpose of this project is to experiment with CUPTI, the Nvidia CUDA Profiling Tools Interface.

CUPTI is a dynamic C/C++ library that can be empowers CUDA developers to trace and profile their GPU code.

## Build

Clone the repo and invoke `make`.

This project is Linux-only for now. My platform is Ubuntu running on Windows Subsystem for Linux 2 (WSL2).

The full CUDA toolkit will need to be [downloaded](https://developer.nvidia.com/cuda-toolkit) and installed--it includes CUPTI (but be aware that CUPTI sometimes releases between CUDA releases, so you might like to update CUPTI to its own latest release).

## Code

### Fairly routine: `main` in `main.cpp`

The `main.cpp` file interprets the command line arguments to specify the number of data points (doubles) in the data set, the size of each block, and the number of steps to use, and which CUDA device to invoke for the calculations.

The number of steps is how far apart the data points will be for consecutive (in the sense of their thread ID) threads. The idea is that certain spacings should cause some number of threads in a warp to access the same bank in the shared memory.

The main function then invokes [`devices_info`](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L170) to iterate all CUDA devices in the system and ouput properties of those devices.

Finally, the main function invokes [`run_calculations`](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L212) to run and profile the CUDA code.

### The interesting stuff: `run_calculations` in `cuda_cupti.cu`

Some block calculations determine how many bloacks are needed to (mostly) evenly distribute the data across blocks of the specified size.

The the [warm-up](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L85) kernel is [invoked](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L228) to warm up the GPU.

Space is [allocated](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L231) on the device for the points data and our "meaningless but verifiable" data is [copied](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L232) there.

The CUPTI API is [invoked](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L234) to request information on kernel and shared memory activity, and to registers callbacks [cuptiBufferRequested](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L29) and [cuptiBufferCompleted](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L40) for CUPTI to use to receive a buffer for recording activity, and to indicate when profiling is complete and the buffer can be harvested for the profiled information.

This is a good place to mention that I have macros to wrap calls to the [CUDA](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L9) and [CUPTI](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L18) APIs. Calls to these APIs should not result in any errors; the macros are designed to detect error conditions on API invocation and end the program if they occur, with some helpful debug output.

The kernel is [invoked](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L242) to run the calculations. There are [multiple](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L95) [versions](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L106) of the kernel in the code that represent the increasing complexity of the kernel. The final version is [`calculate_with_step_shared_mem`](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L118) which copies the data to shared memory and incorporates a step into the index calculation. Again, the idea is that with the right step size bank conflicts will occur.

Calls are [made](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L244) to CUDA to synchronize with the device (that is, wait for it to complete the kernal execution and go idle) and check for any errors, and to CUPTI to flush all recorded activity to the buffer (indicating we're done profiling). A CUDA [call](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L249) copies the results of the calculations back to the host.

Flushing the CUPTI activity buffer causes the [cuptiBufferCompleted](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L40) callback to execute, and that function outputs the profiling information CUPTI gathered.

The data calculations done in the kernel were meaningless but they were deterministic; the result values can now be [checked](https://github.com/dinochiti/cupti-cpp/blob/main/src/cuda_cupti.cu#L253) to confirm that every data point in the data set was modified as expected.

All that's left is to clean up the allocated memory (both device and host) and reset the device.

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
