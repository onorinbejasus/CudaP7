
#include "decompressor.hpp"
#include "common/util/zcompress.hpp"
#include <iostream>
#include <string.h>


namespace aed {
namespace frontend {
namespace stage {

Result Decompressor::initialize()
{
    return RV_OK;
}

void Decompressor::finalize()
{
}

Result Decompressor::process( const DecompressorInput& request )
{
    common::util::ManagedTimer decompressorTimer(
            feconfig.decompressorTimerHandle, 
            request.header.fe_ident);

    common::pipeline::QueuePusher<DecompressorOutput> handle(queue);
    DecompressorOutput& result = *handle;

    result.type = CompositorInput::CI_ResultData;
    result.result_data.header = request.header;

    int ret = common::util::zcompress_inflate(
        request.data, result.result_data.data, request.header.data_length, &result.result_data.header.data_length, MAX_BUFFER_SIZE ); 

    if ( Z_OK != ret ) {
        common::util::zerr(ret); 
        return RV_UNSPECIFIED;
    }

    handle.push();

    return RV_OK;
}

} // stage
} /* frontend */
} /* aed */

