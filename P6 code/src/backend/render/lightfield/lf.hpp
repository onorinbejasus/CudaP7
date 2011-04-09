
#ifndef AED_BACKEND_RENDER_LIGHTFIELD_HPP
#define AED_BACKEND_RENDER_LIGHTFIELD_HPP

#include "backend/render/render_algorithm.hpp"
#include "lightfield.hpp"

namespace aed {
namespace backend {
namespace render {
namespace lightfield {

class LightfieldRenderAlgorithm : public IRenderAlgorithm
{
public:

    LightfieldRenderAlgorithm( const ::lightfield::LFData& lfdata ) : lfdata( lfdata ) { } 
    virtual ~LightfieldRenderAlgorithm() { }

    virtual Result initialize( size_t config_index );
    virtual void destroy();
    virtual Result begin_update( bool* is_update_done, real_t ) { *is_update_done = true; return RV_OK; }
    virtual Result process_data( bool* is_update_done, const RenderableData& ) { *is_update_done = true; return RV_OK; }

    virtual void update_transform_data( const TransformData& t_data ) { }

    virtual RenderResultType get_result_type() const { return RRT_IMAGE; }
    virtual Result query_result( uint8_t*, size_t*, size_t ) { return RV_UNIMPLEMENTED; }
    virtual Result render( uint8_t* color_buffer, uint8_t* depth_buffer, const TransformData& transform );
    
    virtual void update_input_data( MouseData& mouseData ) { }

private:
	const ::lightfield::LFData& lfdata;
	::lightfield::ILightfield* lf;
	uint8_t* color_data;
};

} // namespace lightfield
} // namespace render
} // namespace backend
} // namespace aed

#endif

