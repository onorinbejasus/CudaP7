# cuda headers and library paths
CUDA_H_PATH = /usr/local/cs/cuda/include
CUDA_LIB_PATH = /usr/local/cuda/lib64
CUTIL_H_PATH = $(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc
CUTIL_LIB_PATH = $(HOME)/NVIDIA_GPU_Computing_SDK/C/lib
CUDA_GL_PATH = $(HOME)/NVIDIA_GPU_Computing_SDK/shared/inc

# gcc compiler linker flags
GCC_CFLAGS = -O3
GCC_LFLAGS = -I$(CUDA_GL_PATH) -lGL -lglut -lGLEW

# nvcc compiler/linker flags
NVCC_CFLAGS = -m64 -I$(CUDA_H_PATH) -I$(CUTIL_H_PATH) -L$(CUDA_LIB_PATH) -L$(CUTIL_LIB_PATH) -lcuda -lcudart -lcutil_x86_64 -Xcompiler $(GCC_CFLAGS) $(GCC_LFLAGS)
