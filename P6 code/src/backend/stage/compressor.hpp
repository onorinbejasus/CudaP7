

#ifndef AED_BACK_PLANE_COMPRESSOR_HPP
#define AED_BACK_PLANE_COMPRESSOR_HPP

#include "intermediate_result.hpp"
#include "common/pipeline/pipeline.hpp"

namespace aed {
namespace backend {
namespace stage {

// For lazy programmers like me. 
// The prefix C* is abbreviation of Compressor
typedef IntermediateResult  CompressorRequestData;
typedef IntermediateResult  CompressorResultData;

// is thread safe, can be used in multiple stages on multiple threads.
class Compressor
{
public:
    Compressor( common::pipeline::PipelineQueue<CompressorResultData>& queue )
        : queue(queue) { }
    ~Compressor() {}

    Result initialize() { return RV_OK; }
    void finalize() { }
    Result process( const CompressorRequestData& request ) const;

private:
    common::pipeline::PipelineQueue<CompressorResultData>& queue;

};


} // stage
} // namespace backend
} // namespace aed

#endif // AED_BACK_PLANE_COMPRESSOR_HPP


