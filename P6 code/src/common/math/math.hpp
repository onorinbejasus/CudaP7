/**
 * @file math.hpp
 * @brief General math declarations and definitions.
 *
 * @author Eric Butler (edbutler)
 * @author Zeyang Li (zeyangl)
 */

#ifndef _AED_MATH_MATH_HPP_
#define _AED_MATH_MATH_HPP_

#include <algorithm>
#include <cmath>

namespace aed {

// floating point precision set by this typedef
typedef double real_t;

class Color3;

// since the standard library happily does not provide one
#define PI 3.141592653589793238

/// clamps val to be within the range [min,max].
template<typename T>
inline T clamp( T val, T min, T max )
{
    return std::min( max, std::max( min, val ) );
}

/**
 * Solves the equation a*x*x + b*x + c = 0 for x. Solutions are stored in
 * sol1 and sol2.
 * @return The number of solutions.
 */
unsigned quadratic_formula( real_t a, real_t b, real_t c, real_t* sol1, real_t* sol2 );

template<typename T>
T interpolate( const T& a, const T& b, const T& c, real_t beta, real_t gamma )
{
    real_t alpha = 1 - beta - gamma;
    return alpha * a + beta * b + gamma * c;
}

template<typename T>
inline T interpolate( const T arr[3], real_t beta, real_t gamma )
{
    return interpolate( arr[0], arr[1], arr[2], beta, gamma );
}



template<typename T>
inline bool in_halfopen_range(T val, T min, T max)
{
    return ((min <= val) && (val < max));
}
} /* aed */

#endif /* _AED_MATH_MATH_HPP_ */

