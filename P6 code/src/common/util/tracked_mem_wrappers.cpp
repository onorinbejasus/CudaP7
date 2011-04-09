#include <stdlib.h>
#include "common/util/tracked_mem_wrappers.hpp"

namespace aed {
namespace common {
namespace util {

// XXX what if one one of these allocations fails? what happens to the timer map?

void* tracked_malloc(size_t size, TimerTypeHandle type)
{
    void* ptr = malloc(size);
    TIMER_MANAGER_OBJ.addMemoryData(ptr, size, type);
    return ptr;
}

void tracked_free(void* ptr)
{
    TIMER_MANAGER_OBJ.freeMemory(ptr);
    free(ptr);
}

cudaError_t tracked_cudaMalloc(void** ptr, size_t size, TimerTypeHandle type)
{
    cudaError_t rv = cudaMalloc(ptr, size);
    TIMER_MANAGER_OBJ.addMemoryData(*ptr, size, type);
    return rv;
}

cudaError_t tracked_cudaFree(void* ptr)
{
    TIMER_MANAGER_OBJ.freeMemory(ptr);
    return cudaFree(ptr);
}

/*
void* operator new (size_t size)
{
}


void operator delete (void* ptr)
{
}
*/



}
}
}

