
TARGET = backend
TARGET_TYPE = executable

SRCS = \
	backend/stage/request_receiver.cpp \
	backend/stage/renderable.cpp \
	backend/stage/compressor.cpp \
	backend/stage/result_sender.cpp \
	backend/render/lightfield/lf.cpp \
	texture.cpp \
	backend/main.cpp

CUDA_SRCS = \
	lightfield.cu

COMPILER_FLAGS =
LINK_LIBRARIES = -lpng -lz -lcudart -lboost_thread$(BOOST_LIB_SUFFIX) -lcommon

