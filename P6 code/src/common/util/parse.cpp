
#include "parse.hpp"
#include "common/result.hpp"

namespace aed {
namespace common {
namespace util {

int32_t i32_from_str( const char*           str )
{
    assert( str );
    int x;
    if ( 1 != sscanf( str, "%d", &x ) ) {
        throw ParseErrorException();
    }
    return (int32_t) x;
}

int32_t i32_from_str( const std::string&    str )
{
    return i32_from_str( str.c_str() );
}

double  f64_from_str( const char*           str )
{
    assert( str );
    double x;
    if ( 1 != sscanf( str, "%lf", &x ) ) {
        throw ParseErrorException();
    }
    return x;
}

double  f64_from_str( const std::string&    str )
{
    return f64_from_str( str.c_str() );
}

bool str_ends_with( const char* str, const char* suffix )
{
    assert( str && suffix );

    size_t suf_len = strlen( suffix );
    size_t str_len = strlen( str );

    if ( suf_len > str_len )
        return false;

    size_t idx = str_len - suf_len;
    return strcmp( &str[idx], suffix ) == 0;
}

} // util
} // common
} // aed

