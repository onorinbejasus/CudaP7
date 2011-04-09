#include "common/math/box.hpp"

namespace aed {

BoundingBox3 BoundingBox3::from_center_size( const Vector3& center, const Vector3& size )
{
    Vector3 half_size = size * 0.5;
    return BoundingBox3( center - half_size, center + half_size );
}

Vector3 BoundingBox3::size() const
{
    return max - min;
}

Vector3 BoundingBox3::center() const
{
    return ( max + min ) * 0.5;
}

void BoundingBox3::merge_with( const Vector3& v )
{
    if ( extent == EX_NULL ) {
        max = min = v;
        extent = EX_FINITE;
    } else if ( extent == EX_FINITE ) {
        max = vmax( max, v );
        min = vmin( min, v );
    }
    // else if this->extent == infinite, nothing changes
}

bool BoundingBox3::ray_intersect( const Ray& ray, real_t* enter, real_t* exit ) const
{
    // TODO this could possibly stand for optimization, and definitely
    // could stand for some documentation as to what the heck it's doing
    real_t inv;
    Vector3 e = ray.base;
    Vector3 d = ray.dir;

    if (extent == EX_NULL)
        return false;
    if (extent == EX_INFINITE)
        return true;

    inv = 1 / d[0];
    real_t xmin = ( min[0] - e[0] ) * inv;
    real_t xmax = ( max[0] - e[0] ) * inv;

    real_t txmin, txmax;
    if ( xmin < xmax ) {
        txmin = xmin;
        txmax = xmax;
    } else {
        txmin = xmax;
        txmax = xmin;
    }

    inv = 1 / d[1];
    real_t ymin = ( min[1] - e[1] ) * inv;
    real_t ymax = ( max[1] - e[1] ) * inv;

    real_t tymin, tymax;
    if ( ymin < ymax ) {
        tymin = ymin;
        tymax = ymax;
    } else {
        tymin = ymax;
        tymax = ymin;
    }

    if ( txmin > tymax || tymin > txmax )
        return false;

    inv = 1 / d[2];
    real_t zmin = ( min[2] - e[2] ) * inv;
    real_t zmax = ( max[2] - e[2] ) * inv;

    real_t tzmin, tzmax;
    if ( zmin < zmax ) {
        tzmin = zmin;
        tzmax = zmax;
    } else {
        tzmin = zmax;
        tzmax = zmin;
    }

    if ( txmin > tzmax || tzmin > txmax ||
         tymin > tzmax || tzmin > tymax )
        return false;

    real_t start = std::max(std::max(txmin,tymin),tzmin);
    real_t end = std::min(std::min(txmax,tymax),tzmax);

    if (end <= ray.mint || start >= ray.maxt)
        return false;

    // first intersection occurs once it's crossed all planes
    *enter = std::max(start, ray.mint);
    // last intersection occurs when it leaves the first plane
    *exit = std::min(end, ray.maxt);

    return true;
}

} /* aed */

