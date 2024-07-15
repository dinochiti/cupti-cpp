#ifndef _BC_CUDA_H
#define _BC_CUDA_H

void devices_info(size_t total_data_bytes);
void run_calculations(size_t num_points, size_t block_size, size_t step_size, int dev_num);

#endif // #ifndef _BC_CUDA_H
