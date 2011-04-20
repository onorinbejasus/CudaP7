#ifndef CUDA_ERR_HH
#define CUDA_ERR_HH

#include <cuda.h>

void exit_on_err(cudaError_t err);

#endif /* end of include guard: CUDA_ERR_HH */