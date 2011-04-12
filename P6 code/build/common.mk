
TARGET = libcommon.a
TARGET_TYPE = static_library

SRCS = \
	common/log.cpp \
	common/render_types.cpp \
	common/result.cpp \
	common/math/box.cpp \
	common/math/color.cpp \
	common/math/vector.cpp \
	common/math/matrix.cpp \
	common/math/quaternion.cpp \
	common/math/camera.cpp \
	common/math/camera_roam.cpp \
    common/math/keyframecurve.cpp \
	common/network/socket.cpp \
    common/pipeline/queue.cpp \
	common/util/config_reader.cpp \
	common/util/fixed_pool.cpp \
	common/util/imageio.cpp \
	common/util/parse.cpp \
	common/util/zcompress.cpp \
	common/util/timer.cpp \
	common/util/tracked_mem_wrappers.cpp \
	tinyxml/tinyxml.cpp \
	tinyxml/tinyxmlerror.cpp \
	tinyxml/tinyxmlparser.cpp

COMPILER_FLAGS =

