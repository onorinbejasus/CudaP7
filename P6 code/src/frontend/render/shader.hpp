
#ifndef AED_FRONTEND_RENDER_SHADER_HPP
#define AED_FRONTEND_RENDER_SHADER_HPP

#include "frontend/opengl.hpp"

namespace aed {
namespace frontend {
namespace render {

class ShaderProgram
{
public:
    ShaderProgram() : _program(0)  { }
    ~ShaderProgram() { destroy_shaders(); }

    bool init_shader( const char* vert_file, const char* frag_file );
    bool init_shader(
        const char* vert_file, const char* geom_file, const char* frag_file, 
        GLenum geomInputType, GLenum geomOutputType);
    bool use_shader();
    bool unuse_shader();

    bool self_check();

    void destroy_shaders();
    GLint get_uniform_location(const char* uniform_name);

private:
    GLuint _program;
};

} // render
} /* frontend */
} /* aed */

#endif /* AED_SHADER_HPP */

