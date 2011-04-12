#include "compressor.hpp"
#include "common/log.hpp"
#include "common/util/zcompress.hpp"
#include "common/util/imageio.hpp"
#include <cassert>

namespace aed {
namespace backend {
namespace stage {

Result Compressor::process( const CompressorRequestData& request ) const
{
    common::pipeline::QueuePusher<CompressorResultData> handle(queue);
    CompressorResultData& result = *handle;

    result.header = request.header;
    result.socket = request.socket;

    int ret = common::util::zcompress_deflate( request.data, result.data, request.header.data_length, &result.header.data_length, MAX_BUFFER_SIZE, 7 );
    if (ret != Z_OK)
        common::util::zerr(ret); 

    handle.push();

    return RV_OK;
}

} // stage
} // namespace backend
} // namespace aed

