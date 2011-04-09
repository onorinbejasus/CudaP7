#ifndef AED_BACKEND_RENDER_RENDERALGORITHM_HPP
#define AED_BACKEND_RENDER_RENDERALGORITHM_HPP

#include "common/math/mouse.hpp"
#include "common/result.hpp"
#include "common/network/packet.hpp"
#include "common/network/socket.hpp"
#include <stdint.h>

namespace aed {
namespace backend {
namespace render {

struct RenderableData
{
    common::network::Socket* source;
    objid_t type;
    objid_t stage;
    void* data;
    size_t len;
};

enum RenderResultType
{
    RRT_IMAGE,
    RRT_DATA,
};

class IRenderAlgorithm
{
public:
    virtual ~IRenderAlgorithm() {}
    virtual Result initialize( size_t config_index ) = 0;
    virtual void destroy() = 0;
    virtual Result begin_update( bool* is_update_done, real_t delta_seconds ) = 0;
    // the data pointer is only valid until the function returns
    virtual Result process_data( bool* is_update_done, const RenderableData& data ) = 0;

    virtual void update_transform_data( const TransformData& t_data ) = 0;

    virtual RenderResultType get_result_type() const = 0;
    virtual Result query_result( uint8_t* data, size_t* len, size_t max_len ) = 0; // for RRT_DATA
    virtual Result render( uint8_t* color_buffer, uint8_t* depth_buffer, const TransformData& transform ) = 0; // for RRT_IMAGE

    virtual void update_input_data( MouseData& mouseData ) = 0;
};

}
}
}

#endif

