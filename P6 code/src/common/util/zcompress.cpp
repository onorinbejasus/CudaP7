#include "zcompress.hpp"
#include "common/log.hpp"

namespace aed {
namespace common {
namespace util {

int zcompress_deflate(const unsigned char *in, unsigned char *out, size_t in_size, size_t* out_size, size_t buffer_size, int level)
{
    int ret, flush;
    z_stream strm;

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, level);
    if (ret != Z_OK)
        return ret;

    strm.avail_in = in_size;
    flush = Z_FINISH;
    strm.next_in = (unsigned char*)in;
 
    strm.avail_out = buffer_size;
    strm.next_out = out;
    ret = deflate(&strm, flush);    /* no bad return value */
    assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
    *out_size = buffer_size - strm.avail_out;

    assert(strm.avail_in == 0);     /* all input will be used */
    assert(ret == Z_STREAM_END);    /* stream will be complete */

    /* clean up and return */
    (void)deflateEnd(&strm);
    return Z_OK;
}

/* Decompress from file source to file dest until stream ends or EOF.
   inf() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_DATA_ERROR if the deflate data is
   invalid or incomplete, Z_VERSION_ERROR if the version of zlib.h and
   the version of the library linked do not match, or Z_ERRNO if there
   is an error reading or writing the files. */
int zcompress_inflate(const unsigned char *in, unsigned char *out, size_t in_size, size_t* out_size, size_t buffer_size )
{
    int ret;
    z_stream strm;

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);
    if (ret != Z_OK)
        return ret;

    /* decompress until deflate stream ends or end of file */
    strm.avail_in = in_size;
    if (strm.avail_in == 0)
        return Z_OK;
    strm.next_in = (unsigned char*)in;
    /* run inflate() on input until output buffer not full */
        
    strm.avail_out = buffer_size;
    strm.next_out = out;
    ret = inflate(&strm, Z_NO_FLUSH);
    assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
    switch (ret) {
        case Z_NEED_DICT:
            ret = Z_DATA_ERROR;     /* and fall through */
        case Z_DATA_ERROR:
        case Z_MEM_ERROR:
            (void)inflateEnd(&strm);
            return ret;
    }
    *out_size = buffer_size - strm.avail_out;
  
    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

/* report a zlib or i/o error */
void zerr(int ret)
{
/*
    fputs("zpipe: ", stderr);
    switch (ret) {
    case Z_ERRNO:
        if (ferror(stdin))
            fputs("error reading stdin\n", stderr);
        if (ferror(stdout))
            fputs("error writing stdout\n", stderr);
        break;
    case Z_STREAM_ERROR:
        fputs("invalid compression level\n", stderr);
        break;
    case Z_DATA_ERROR:
        fputs("invalid or incomplete deflate data\n", stderr);
        break;
    case Z_MEM_ERROR:
        fputs("out of memory\n", stderr);
        break;
    case Z_VERSION_ERROR:
        fputs("zlib version mismatch!\n", stderr);
    }
*/
    switch (ret) {
    case Z_ERRNO:
        if (ferror(stdin))
            LOG_MSG(MDL_COMPRESSION, SVR_CRITICAL, "ZCompress: error reading stdin");
        if (ferror(stdout))
            LOG_MSG(MDL_COMPRESSION, SVR_CRITICAL, "ZCompress: error writing stdout");
        break;
    case Z_STREAM_ERROR:
        LOG_MSG(MDL_COMPRESSION, SVR_CRITICAL, "ZCompress: invalid compression level");
        break;
    case Z_DATA_ERROR:
        LOG_MSG(MDL_COMPRESSION, SVR_CRITICAL, "ZCompress: invalid or incomplete deflate data");
        break;
    case Z_MEM_ERROR:
        LOG_MSG(MDL_COMPRESSION, SVR_CRITICAL, "ZCompress: out of memory");
        break;
    case Z_VERSION_ERROR:
        LOG_MSG(MDL_COMPRESSION, SVR_CRITICAL, "ZCompress: zlib version mismatch!");
        break;
    default:
        LOG_MSG(MDL_COMPRESSION, SVR_CRITICAL, "ZCompress: found an unknown error code");
    }
}

} // util
} // common
} // namespace aed

