
#ifndef AED_STAGE_COMPOSITOR_HPP
#define AED_STAGE_COMPOSITOR_HPP

#include "frontend/config.hpp"
#include "common/util/fixed_pool.hpp"
#include <cassert>

#ifdef AEDGL
#include "frontend/opengl.hpp"
#include "frontend/render/shader.hpp"
#include "frontend/render/lightfield_render.hpp"
#include <cuda_runtime.h>
#else

struct float2 { float x; float y; };
struct float3 { float x; float y; float z; };
struct float4 { float x; float y; float z; float w; };

#endif

namespace aed {
namespace frontend {
namespace stage {

// CompositorInput is already declared in config.hpp

struct ReceivedData
{
    bool received_color;
    bool received_depth;
    ReceivedData() : received_color( false ), received_depth( false ) {}
    // XXX uncomment this once depth textures are being sent
    bool received_all() const { return received_color  && received_depth ; }
    bool& operator[]( uint64_t type ) {
        switch ( type ) {
        case BRDT_COLOR_IMAGE: return received_color;
        case BRDT_DEPTH_IMAGE: return received_depth;
        default: assert( false );
        }
    }
};

struct CompositorFrameData
{
    CompositorFrameData () : info(NULL) { early_data.reserve(MAX_OBJECTS); }

    ReceivedData received_table[MAX_OBJECTS];
    FrameInfo* info;
    std::vector<ResultData*> early_data;
};

class Compositor
{
public:
    Compositor( MutexCvarPair* frame_mcp )
      : frame_data( MAX_FRAMES_IN_FLIGHT ),
        current_frame( 0u ),
        frame_mcp( frame_mcp ),
        image_counter( 0 ),
        do_save_images( false )
        {}
    ~Compositor() {}

    Result initialize();
    void finalize();

    Result process( const CompositorInput& request );

    void activate_save_images( size_t start_image );

private:
    typedef std::vector<CompositorFrameData> CFDList;

    Result process_frame_info( FrameInfo* info );
    Result process_object( const ResultData* data );
    Result begin_frame( CompositorFrameData* data );
    Result check_and_render();
#ifdef AEDGL
    void render( const CompositorFrameData& data );
#endif

    CFDList frame_data;
    size_t current_frame;

    MutexCvarPair* frame_mcp;

    size_t image_counter;
    bool do_save_images;

    // storage for data that arrives early
    common::util::FixedPool* storage_pool;

#ifdef AEDGL
    render::LightfieldRenderer lfrender;
#endif
};


} // stage
} /* frontend */
} /* aed */

#endif /* AED_COMPOSITOR_HPP */

