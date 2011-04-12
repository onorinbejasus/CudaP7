
TARGET = lightfield
TARGET_TYPE = executable

SRCS = \
	imageio.cpp \
	application.cpp \
	texture.cpp \
	main.cpp

CUDA_SRCS = \
	lightfield.cu

COMPILER_FLAGS =
LINK_LIBRARIES = -lGL -lSDLmain -lSDL -lpng -lz -lcudart

