
# directories

# source file diretory
SRC_DIR = src
# top-level output directory (for .o and .d files)
OBJ_DIR = obj
# top-level dir for output binaries
BIN_DIR = bin

#find cuda library
OSNAME := $(shell uname -s)
MACHNAME := $(shell uname -n | sed -r 's/.*[0-9]+\.//')
ifeq ($(OSNAME),Darwin)
	include build/makefile.osx
else
ifeq ($(MACHNAME),ghc.andrew.cmu.edu)
CUDA_H_PATH = /usr/local/cs/cuda/include
CUDA_LIB_PATH = /usr/local/cs/cuda/lib64
else
CUDA_H_PATH = /usr/local/cuda/include
CUDA_LIB_PATH = /usr/local/cuda/lib64
endif
endif

# HACK find the boost library
BOOST_LIB_SUFFIX =
ifneq ($(strip $(wildcard /usr/lib/libboost_thread-mt.a)),)
BOOST_LIB_SUFFIX = -mt
endif
ifneq ($(strip $(wildcard /usr/lib64/libboost_thread-mt.so)),)
BOOST_LIB_SUFFIX = -mt
endif

