#ifndef AED_TRACKED_MEM_WRAPPERS_HPP
#define AED_TRACKED_MEM_WRAPPERS_HPP

#include "common/util/timer.hpp"
#include <cuda_runtime.h>

namespace aed {
namespace common {
namespace util {

void* tracked_malloc(size_t size, TimerTypeHandle type);
void tracked_free(void* ptr);

cudaError_t tracked_cudaMalloc(void** ptr, size_t size, TimerTypeHandle type);
cudaError_t tracked_cudaFree(void* ptr);

template<typename T>
T* tracked_new_arr(size_t len, TimerTypeHandle type)
{
    T* arr = new T[len];
    TIMER_MANAGER_OBJ.addMemoryData(arr, len * sizeof(T), type);
    return arr;
}

template<typename T>
void tracked_delete_arr(T* ptr)
{
    TIMER_MANAGER_OBJ.freeMemory(ptr);
    delete[] ptr;
}


}
}
}

#endif // AED_TRACKED_MEM_WRAPPERS_HPP

