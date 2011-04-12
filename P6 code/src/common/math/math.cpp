/**
 * @file math.cpp
 * @brief General math function definitions.
 *
 * @author Eric Butler (edbutler)
 */

#include "common/math/math.hpp"

#ifdef SOLUTION
namespace aed {

unsigned quadratic_formula( real_t a, real_t b, real_t c, real_t* sol1, real_t* sol2 )
{
    const real_t desc = b * b - 4 * a * c;

    if ( desc < 0 )
        return 0;

    real_t inv2a = 1 / ( 2 * a );
    real_t negbo2a = -b * inv2a;

    if ( desc == 0 ) {
        *sol1 = *sol2 = negbo2a;
        return 1;
    }

    real_t descrooto2a = sqrt( desc ) * inv2a;
    *sol1 = negbo2a - descrooto2a;
    *sol2 = negbo2a + descrooto2a;
    return 2;
}

} /* aed */
#endif /* SOLUTION */
