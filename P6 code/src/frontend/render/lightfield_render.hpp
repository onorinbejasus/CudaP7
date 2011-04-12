
#ifndef AED_FRONTEND_RENDER_LIGHTFIELDRENDER_HPP
#define AED_FRONTEND_RENDER_LIGHTFIELDRENDER_HPP

#include "frontend/opengl.hpp"
#include "frontend/config.hpp"

namespace aed {
namespace frontend {
namespace render {

class ShaderProgram;

class LightfieldRenderer
{
public:
    LightfieldRenderer();
    ~LightfieldRenderer();
    void initialize();
    void destroy();
    void begin_frame( const FrameInfo* frame );
    void end_frame();
    void add_texture( BackendResponseDataType type, size_t index, const ClipRect& clip, const ResultData& image );
    void render_pass( size_t pass );

    class Impl;
private:
    Impl* impl;
};

} // render
} /* frontend */
} /* aed */

#endif /* AED_TEXTURE_HPP */

