#include <iostream>

#include "cupti.h"

using std::cout;
using std::cerr;

// wrapper macro for CUDA API calls
#define CUDA_ERR_SAFE(cuda_call) { \
cudaError_t cuda_error_code = (cuda_call); \
if (cudaSuccess != cuda_error_code) { \
    cerr << "CUDA error " << cuda_error_code << ": '" << cudaGetErrorString(cuda_error_code) << "'; line " << __LINE__ << " file " << __FILE__ << "; aborting\n"; \
    exit(EXIT_FAILURE); \
} \
}

// wrapper macro for CUPTI API calls
#define CUPTI_ERR_SAFE(cupti_call) { \
CUptiResult cupti_result = (cupti_call); \
if (cupti_result != CUPTI_SUCCESS) { \
    const char* cupti_err_str; \
    cuptiGetResultString(cupti_result, &cupti_err_str); \
    cerr << "CUPTI error " << cupti_result << ": '" << cupti_err_str << "'; line " << __LINE__ << " file " << __FILE__ << "; aborting\n"; \
    exit(EXIT_FAILURE); \
} \
}

// callback for CUPTI to request a buffer to populate with the requested activity records
void CUPTIAPI cuptiBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    // 16 MB; docs recommend 1 to 10 MB
    //   this is a lot more than needed for this workload though
    *size = 16 * 1024 * 1024;
    *buffer = (uint8_t*) malloc(*size);
    // 0 means CUPTI will return all records (but still won't overflow buffer)
    *maxNumRecords = 0;
}

// callback for CUPTI to invoke when activity recording is complete
//  - enumerates and ouputs the activity records
void CUPTIAPI cuptiBufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    cout << "CUPTI buffer completed\n";

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            // Process record (kernel execution times, etc.)
            if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
                CUpti_ActivityKernel7 *kernel = (CUpti_ActivityKernel7 *)record;
                printf("Kernel execution: %s; grid size (%d,%d,%d); block size (%d,%d,%d); time %llu ns\n",
                    kernel->name,
                    kernel->gridX, kernel->gridY, kernel->gridZ,
                    kernel->blockX, kernel->blockY, kernel->blockZ,
                    (unsigned long long)(kernel->end - kernel->start));
                printf("                  registers per thread  %d\n", kernel->registersPerThread);
                printf("                  static shared memory  %d bytes\n", kernel->staticSharedMemory);
                printf("                  dynamic shared memory %d bytes\n", kernel->dynamicSharedMemory);
                printf("                  local memory per thread %u bytes\n", kernel->localMemoryPerThread);
            }
            else if (record->kind == CUPTI_ACTIVITY_KIND_SHARED_ACCESS) {
                CUpti_ActivitySharedAccess *sharedAccess = (CUpti_ActivitySharedAccess *)record;
                printf("Shared access: number of shared transactions %lu\n", sharedAccess->sharedTransactions);
            }
            else if (record->kind == CUPTI_ACTIVITY_KIND_FUNCTION) {
                CUpti_ActivityModule *module = (CUpti_ActivityModule *)record;
                printf("Function activity: module id %u\n", module->contextId);
            }
            else {
                printf("Record of type %d (not unpacking)\n", record->kind);
            }
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else {
            // here's a neat trick--the macro works just fine if fed the return value
            //   of a CUPTI function call instead of an actual CUPTI function call
            CUPTI_ERR_SAFE( status );
        }
    } while (1);
}

// trivial kernel to invoke to warm up a CUDA device; called before profiling
//   to ensure initialization activities/times don't affect profiling
__global__ void warm_up() {
    // this is not used as an index here, but this is how we would calculate the
    //   index into the data for a one-dimensional array
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    double x1 = 0.0;
    double x2 = 2.0;
    x1 = x1 + (double) index * x2 + (double) threadIdx.x;
}

// simple kernel to run the standard calculation on a single point
__global__ void calculate(double *points, size_t num_points) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_points) {
        points[index] *= static_cast<double>(index);
    }
}

// kernel to run the standard calculation on a single point but with stepping--all
//   points get calculated but the threads in a block will calculate points separated
//   by the step count instead of consecutive points (pass in a step count of 1 to
//   have the same behavior as the plain calculate kernel)
__global__ void calculate_with_step(double *points, size_t step_size, size_t num_points) {
    const unsigned int offset = blockIdx.x % step_size;
    const unsigned int span_num = (blockIdx.x / step_size) * step_size * blockDim.x;
    const unsigned int index =  span_num + threadIdx.x * step_size + offset;
    if (index < num_points) {
        points[index] *= static_cast<double>(index);
    }
}

// kernel to run the standard calculation on a single point but with stepping, and
//   copying the data to shared memory
//  - extra, meaningless calculation steps added to up the number of clock cycles
__global__ void calculate_with_step_shared_mem(double *points, size_t step_size, size_t num_points) {
    extern __shared__ double shared_data[];
    const unsigned int offset = blockIdx.x % step_size;
    const unsigned int span_num = (blockIdx.x / step_size) * step_size * blockDim.x;
    const unsigned int local_index = threadIdx.x * step_size + offset;
    const unsigned int global_index = span_num + local_index;
    if (global_index < num_points) {
        double original_value = points[global_index];
        // arbitrary calculations; hit the shared data multiple times
        shared_data[local_index] = original_value;
        shared_data[local_index] += original_value;
        shared_data[local_index] *= original_value;
        shared_data[local_index] -= original_value;
        shared_data[local_index] *= original_value;
        shared_data[local_index] /= original_value;
        shared_data[local_index] -= original_value;
        shared_data[local_index] /= original_value;
        shared_data[local_index] += original_value;
        shared_data[local_index] *= original_value;
        // final, 'correct' calculation
        shared_data[local_index] = original_value;
        shared_data[local_index] *= static_cast<double>(global_index);
        points[global_index] = shared_data[local_index];
    }
}

// Debug function to output the points data; used to diagnose mismatches
//   is expected and actual calculation outcomes
void data_dump(double *points, size_t num_points) {
    for (size_t index = 0; index < num_points; ++index) {
        if (0 == index % 10) {
            cout << "\n" << points[index];
        }
        else {
            cout << "\t" << points[index];
        }
    }
    cout << "\n\n";
}

// compare 'equality' of doubles for both absolute and relative tolerance
bool doubles_within_tolerance(const double x, const double y) {
    const double tolerance = 0.0000001;
    const double diff = abs(x - y);
    if (diff < tolerance) {
        return true;
    }
    const double maxd = (abs(x) > abs(y)) ? abs(x) : abs(y);
    return diff < tolerance * maxd;
}

// enumerate devices and output some useful properties
void devices_info(size_t total_data_bytes) {
    int cuda_dev_count;
    CUDA_ERR_SAFE( cudaGetDeviceCount(&cuda_dev_count) );
    if (cuda_dev_count < 1) {
        cerr << "No CUDA devices available\n";
        exit(EXIT_FAILURE);
    }

        for (int dev_num = 0; dev_num < cuda_dev_count; ++dev_num) {
        cudaDeviceProp cuda_dev_properties;
        CUDA_ERR_SAFE( cudaGetDeviceProperties(&cuda_dev_properties, dev_num) );
        cout << "\ndevice number: " << dev_num << "\n";
        cout << "device name: " << cuda_dev_properties.name << "\n";
        cout << "total global memory: " << cuda_dev_properties.totalGlobalMem << "\n";
        cout << "memory bus width: " << cuda_dev_properties.memoryBusWidth << "\n";
        cout << "clock rate: " << cuda_dev_properties.clockRate << "\n";
        cout << "multiprocessor count: " << cuda_dev_properties.multiProcessorCount << "\n";
        cout << "max threads per multiprocessor: " << cuda_dev_properties.maxThreadsPerMultiProcessor << "\n";
        cout << "registers per multiprocessor: " << cuda_dev_properties.regsPerMultiprocessor << "\n";
        cout << "warp size: " << cuda_dev_properties.warpSize << "\n";
        cout << "registers per block: " << cuda_dev_properties.regsPerBlock << "\n";
        cout << "shared memory per block: " << cuda_dev_properties.sharedMemPerBlock << "\n";
        cout << "max dimension size of a grid (x, y, z): " 
            << cuda_dev_properties.maxGridSize[0] << ", " 
            << cuda_dev_properties.maxGridSize[1] << ", " 
            << cuda_dev_properties.maxGridSize[2] << "\n";
        cout << "max dimension size of a thread block (x, y, z): " 
            << cuda_dev_properties.maxThreadsDim[0] << ", " 
            << cuda_dev_properties.maxThreadsDim[1] << ", " 
            << cuda_dev_properties.maxThreadsDim[2] << "\n";
        cout << "concurrent kernels: " << cuda_dev_properties.concurrentKernels << "\n";
        cout << "local L1 cache supported: " 
            << (cuda_dev_properties.localL1CacheSupported ? "yes" : "no") << "\n";
        if (cuda_dev_properties.totalGlobalMem < total_data_bytes) {
            cout << "** CUDA device insufficient global memory; need " << total_data_bytes << " bytes for data alone; " << cuda_dev_properties.totalGlobalMem << " on device\n";
        }
    }
}

// execute kernels on the specific device
//  - profiles with CUPTI
//  - checks the data output to ensure every point was properly calculated
void run_calculations(size_t num_points, size_t block_size, size_t step_size, int dev_num) {
    CUDA_ERR_SAFE( cudaSetDevice(dev_num) );
    size_t num_blocks = (num_points + block_size - 1) / block_size;
    // some extra blocks for when step size does not divide evenly into block size, to
    //   ensure the last data points get calculated
    num_blocks += ((num_blocks % step_size) > 0) ? step_size - (num_blocks % step_size) : 0;
    cout << "Allocating and initializing " << num_points << " doubles\n";
    double *host_data = new double[num_points];
    size_t num_point_bytes = num_points * sizeof(double);
    // meaningless but verifiable data: set each element to one-tenth its position in the array
    for (size_t index = 0; index < num_points; ++index) {
        host_data[index] = static_cast<double>(index) / 10.0;
    }
    // data_dump(host_data, num_points); // TODO DEBUG

    // run warm-up before activating CUPTI
    warm_up<<<num_blocks, block_size>>>();

    double *device_data;
    CUDA_ERR_SAFE( cudaMalloc(&device_data, num_point_bytes) );
    CUDA_ERR_SAFE( cudaMemcpy(device_data, host_data, num_point_bytes, cudaMemcpyHostToDevice) );

    CUPTI_ERR_SAFE( cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL) );
    CUPTI_ERR_SAFE( cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SHARED_ACCESS) );
    CUPTI_ERR_SAFE( cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted) );

    CUDA_ERR_SAFE( cudaGetLastError() );

    // Compute!
    cout << "Running " << num_blocks << " blocks of size " << block_size << " with step size " << step_size << "\n";
    calculate_with_step_shared_mem<<<num_blocks, block_size, block_size * sizeof(double) * step_size>>>(device_data, step_size, num_points);

    CUDA_ERR_SAFE( cudaDeviceSynchronize() );
    CUDA_ERR_SAFE( cudaGetLastError() );

    CUPTI_ERR_SAFE( cuptiActivityFlushAll(0) );

    CUDA_ERR_SAFE( cudaMemcpy(host_data, device_data, num_point_bytes, cudaMemcpyDeviceToHost) );

    // check computation correctness; verifies indexes were properly calculated in the kernel(s)
    // data_dump(host_data, num_points); // TODO DEBUG
    for (size_t index = 0; index < num_points; ++index) {
        const double original_value = static_cast<double>(index) / 10.0;
        const double new_value = host_data[index];
        const double expected_value = (original_value * original_value * 10.0);
        if ( !doubles_within_tolerance(new_value, expected_value) ) {
            cerr << "** DATA ERROR **: Result data incorrect at index " << index
                 << "; expected " << (original_value * original_value * 10.0)
                 << " and encountered " << host_data[index] << "\n";
            // only need to report a single incorrect value
            break;
        }
    }

    CUDA_ERR_SAFE( cudaFree(device_data) );
    delete[] host_data;
    CUDA_ERR_SAFE( cudaDeviceReset() );
}
