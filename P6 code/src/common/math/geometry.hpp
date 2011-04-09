
#ifndef AED_COMMON_MATH_GEOMETRY_HPP
#define AED_COMMON_MATH_GEOMETRY_HPP

#include "common/math/camera.hpp"
#include <stdint.h>

namespace aed {

struct ClipRect
{
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

struct Transform
{
    Vector3 position;
    Quaternion orientation;
    Vector3 scale;
};

}

#endif

