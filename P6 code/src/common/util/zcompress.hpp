/** @file zcompress.hpp
 *
 *  This file using the the zlib compression library to make stream compression
 *  mechanics, to shorten the data transfer time from the bandwidth limitation 
 *
 */

#ifndef _ZCOMPRESS_H_
#define _ZCOMPRESS_H_

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <zlib.h>

namespace aed {
namespace common {
namespace util {

int zcompress_deflate(const unsigned char *in, unsigned char *out, size_t in_size, size_t* out_size, size_t buffer_size, int level);

int zcompress_inflate(const unsigned char *in, unsigned char *out, size_t in_size, size_t* out_size, size_t buffer_size );

void zerr(int ret);

} // util
} // common
} // namespace aed

#endif /* _ZCOMPRESS_H_ */
