
TARGET = frontend
TARGET_TYPE = executable

SRCS = \
	frontend/main.cpp \
	frontend/frontplane.cpp \
	frontend/player.cpp \
	frontend/stage/analyzer.cpp \
	frontend/stage/receiver.cpp \
	frontend/stage/decompressor.cpp \
	frontend/stage/compositor.cpp \
    frontend/render/image_save.cpp \
	frontend/render/shader.cpp \
	frontend/render/lightfield_render.cpp \
	frontend/render/mesh.cpp \
	frontend/render/material.cpp 

COMPILER_FLAGS = -DAEDGL
LINK_LIBRARIES = -lGLEW -lGL -lGLU -lSDLmain -lSDL -lpng -lz -lboost_thread$(BOOST_LIB_SUFFIX) -lcommon

