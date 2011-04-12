
#ifndef AED_COMMON_UTIL_PARSE_HPP
#define AED_COMMON_UTIL_PARSE_HPP

#include <string>
#include <stdint.h>

namespace aed {
namespace common {
namespace util {

int32_t i32_from_str( const char*           str );
int32_t i32_from_str( const std::string&    str );
double  f64_from_str( const char*           str );
double  f64_from_str( const std::string&    str );

bool str_ends_with( const char* str, const char* suffix );

} // util
} // common
} // aed

#endif

