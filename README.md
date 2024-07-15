# CUPTI and CUDA

The purpose of this project is to experiment with CUPTI, the Nvidia CUDA Profiling Tools Interface.

CUPTI is a dynamic C/C++ library that can be empowers CUDA developers to trace and profile their GPU code.

## Build

Clone the repo and invoke `make`.

This project is Linux-only for now. My platform is Ubuntu running on Windows Subsystem for Linux 2 (WSL2).

The full CUDA toolkit will need to be [downloaded](https://developer.nvidia.com/cuda-toolkit) and installed--it includes CUPTI (but be aware that CUPTI sometimes releases between CUDA releases, so you might like to update CUPTI to its own latest release).
