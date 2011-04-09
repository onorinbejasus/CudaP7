#ifndef AED_FRONTEND_STAGE_DECOMPRESSOR_HPP
#define AED_FRONTEND_STAGE_DECOMPRESSOR_HPP

#include "common/pipeline/pipeline.hpp"
#include "frontend/config.hpp"

namespace aed {
namespace frontend {
namespace stage {

typedef ResultData DecompressorInput;
typedef CompositorInput DecompressorOutput;

/*
class Decompressor { use new pipeline stuff }
*/

class Decompressor
{
public:
    Decompressor(common::pipeline::PipelineQueue<DecompressorOutput>& in_queue)
        : queue(in_queue) {}
    ~Decompressor() {}

    Result initialize();
    void finalize();

	Result process( const DecompressorInput &request );

private:
	bool decompress( ProcessingImage &processing_image );
    common::pipeline::PipelineQueue<DecompressorOutput>& queue;
};

} // stage
} /* frontend */
} /* aed */

#endif /* AED_DECOMPRESSOR_HPP */

