#ifndef _AED_MATH_BOX_HPP_
#define _AED_MATH_BOX_HPP_

#include "common/math/vector.hpp"

namespace aed {

struct Ray
{
    Vector3 base;
    Vector3 dir;
    /// time bounds [min_time, max_time).
    real_t mint, maxt;

    Ray() { }

    Ray( const Vector3& b, const Vector3& d, real_t mn, real_t mx )
        : base( b ), dir( d ), mint( mn ), maxt( mx ) { }
};

class BoundingBox3
{
public:

    enum Extent
    {
        /**
         * Indicates box takes up no space at all. Never intersects with
         * another box and is enclosed by everything.
         */
        EX_NULL,

        /**
         * Indicates box is finite. Position given by min and max.
         */
        EX_FINITE,

        /**
         * Indictes box is infinite. Intersects everything and ecloses
         * everything.
         */
        EX_INFINITE
    };

    // the extents of the box
    Vector3 min, max;
    // it's extent value
    Extent extent;

    static BoundingBox3 from_center_size( const Vector3& center, const Vector3& size );

    // leaves values uninitialized
    BoundingBox3() { }

    /**
     * Leaves min/max unitialized
     */
    explicit BoundingBox3( Extent e )
        : extent( e ) { }

    // from corners
    BoundingBox3( const Vector3& min, const Vector3& max )
        : min( min ), max( max ), extent( EX_FINITE ) { }


    // access functions

    Vector3 size() const;

    Vector3 center() const;

    // mutation functions

    /**
     * Unions this box with the given vector. The result will emcompass this box
     * and the given vector as tightly as possible.
     */
    void merge_with( const Vector3& v );

    // intersection tests

    /**
     * Returns true iff the given ray intersects with this box.
     * @param ray The ray.
     * @param enter The time of entry is stored here. Must be non-null.
     * @param enter The time of exit is stored here. Must be non-null.
     * @return true iff there is an intersection.
     */
    bool ray_intersect( const Ray& ray, real_t* enter, real_t* exit ) const;
};

} /* aed */

#endif /* _AED_MATH_BOX_HPP_ */

