#ifndef AED_BACK_PLANE_RENDERABLE_HPP
#define AED_BACK_PLANE_RENDERABLE_HPP

#include "intermediate_result.hpp"
#include "common/pipeline/pipeline.hpp"
#include "backend/render/render_algorithm.hpp"

namespace aed {
namespace backend {
namespace stage {

typedef BackendRequestPacket    RenderableRequestData;
typedef IntermediateResult      RenderableResultData;

class Renderable
{
public:
    Renderable( render::IRenderAlgorithm* render_algorithm, common::pipeline::PipelineQueue<RenderableResultData>& queue, size_t config_index )
        : render_algorithm(render_algorithm), queue(queue), config_index(config_index), is_rendering_request( false ) { }
    ~Renderable() {}

    Result initialize();
    void finalize();

    Result process( const RenderableRequestData& request );

private:

    render::IRenderAlgorithm* render_algorithm;
    common::pipeline::PipelineQueue<RenderableResultData>& queue;
    // Index of this Renderable in configuration settings
    size_t config_index;

    BackendRenderingRequest render_request;
    bool is_rendering_request;
    common::network::Socket* request_source;
};

class PhonyRenderable : public render::IRenderAlgorithm
{
public:
    virtual ~PhonyRenderable() {}
    virtual Result initialize( size_t ) { return RV_OK; }
    virtual void destroy() { }
    virtual Result begin_update( bool* is_update_done, real_t ) { *is_update_done = true; return RV_OK; }
    virtual Result process_data( bool* is_update_done, const render::RenderableData& ) { *is_update_done = true; return RV_OK; }

    virtual render::RenderResultType get_result_type() const { return render::RRT_IMAGE; }
    virtual Result query_result( uint8_t*, size_t*, size_t ) { return RV_UNIMPLEMENTED; }
    virtual Result render( uint8_t*, uint8_t*, const TransformData& ) { return RV_OK; }
};


} // stage
} // namespace backend
} // namespace aed

#endif // AED_BACK_PLANE_RENDERABLE_HPP


