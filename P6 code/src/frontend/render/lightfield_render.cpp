#include "lightfield_render.hpp"
#include "common/util/imageio.hpp"
#include "common/util/tracked_mem_wrappers.hpp"
#include "common/math/camera.hpp"
#include "frontend/render/shader.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

namespace aed {
namespace frontend {
namespace render {

enum LFRState
{
    // before initialization
    LFRS_UNINITIALIZED,
    // after initialization, but not during a frame
    LFRS_READY,
    // between begin/end frame calls
    LFRS_FRAME_BEGUN,
};

class LightfieldRenderer::Impl
{
public:
    LFRState state;

    GLuint color_texname;
    GLuint depth_texname;
    unsigned char* black_screen;

    ShaderProgram shader;

    // the most recent frame info
    const FrameInfo* info;
    float distance[MAX_OBJECTS];
};

LightfieldRenderer::LightfieldRenderer()
{
    impl = new Impl();
    impl->state = LFRS_UNINITIALIZED;
}

LightfieldRenderer::~LightfieldRenderer()
{
    destroy();
    delete impl;
}

// leaves null texture bound
static GLuint create_2darray_texture( GLenum format, size_t width, size_t height, size_t depth )
{
    GLuint tex;

    glGenTextures( 1, &tex );
    if ( tex == 0 ) { throw ResourceUnavailableException(); }

    glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, tex );
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

    glTexImage3D(
        GL_TEXTURE_2D_ARRAY_EXT, 0, format,
        width, height, depth,
        0, format, GL_UNSIGNED_BYTE, NULL );
    // XXX no error checking

    glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, 0 );

    return tex;
}

void LightfieldRenderer::initialize() {	
    assert( impl->state == LFRS_UNINITIALIZED );

    // XXX doesn't properly clean up on failure

    size_t result_data_buffer_size = MAX_BUFFER_SIZE;
    //impl->black_screen = new unsigned char[result_data_buffer_size];
    impl->black_screen = common::util::tracked_new_arr<unsigned char>(result_data_buffer_size, feconfig.feMemoryHandle);
	memset( impl->black_screen, 0, result_data_buffer_size );

    impl->color_texname = create_2darray_texture( GL_RGBA, feconfig.window_width, feconfig.window_height, MAX_OBJECTS );
    impl->depth_texname = create_2darray_texture( GL_LUMINANCE, feconfig.window_width, feconfig.window_height, MAX_OBJECTS );

    if ( !impl->shader.init_shader("shaders/compositor_vert.glsl", "shaders/compositor_frag.glsl") ) {
        throw UnspecifiedException();
    }

    impl->state = LFRS_READY;
}

void LightfieldRenderer::destroy()
{
    if ( impl->state == LFRS_UNINITIALIZED ) { return; }

    //assert( impl->state == LFRS_READY );
    // TODO unimplemented
}

void LightfieldRenderer::begin_frame( const FrameInfo* frame )
{
    assert( impl->state == LFRS_READY );

    // XXX no state management to ensure begin/end were called
    // TODO count how many lfs there are and figure out what their indices are for each pass

    size_t num_objects = frame->objects.size();
    for ( size_t index = 0; index < num_objects; index++ ) {
        //objectid[index] = geometry->get_geometry_type_id( cfdata.info->objects[index].object );
        impl->distance[index] = length( frame->camera.get_position() - frame->objects[index].transform.position ); 
    }

    impl->info = frame;
    impl->state = LFRS_FRAME_BEGUN;
}

void LightfieldRenderer::end_frame()
{
    assert( impl->state == LFRS_FRAME_BEGUN );
    // TODO unimplemented
    impl->state = LFRS_READY;
}

void LightfieldRenderer::add_texture( BackendResponseDataType type, size_t index, const ClipRect& clip, const ResultData& image )
{
    assert( impl->state == LFRS_FRAME_BEGUN );

    switch ( type ) {
    case BRDT_COLOR_IMAGE:
    {
        glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, impl->color_texname);
        
        glTexSubImage3D(
            GL_TEXTURE_2D_ARRAY_EXT, 0, 
            0, 0, index, clip.width, clip.height,
            1, GL_RGBA, GL_UNSIGNED_BYTE, image.data);
    } break;
    case BRDT_DEPTH_IMAGE:
    {
        glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, impl->depth_texname);
        
        glTexSubImage3D(
            GL_TEXTURE_2D_ARRAY_EXT, 0, 
            0, 0, index, clip.width, clip.height,
            1, GL_LUMINANCE, GL_UNSIGNED_BYTE, image.data);
    } break;
    case BRDT_DATA:
    {
        throw InvalidArgumentException("add_texture received data of type BRDT_DATA"); 
    } break;
    }
}

void LightfieldRenderer::render_pass( size_t pass )
{
    assert( impl->state == LFRS_FRAME_BEGUN );

    // set up camera
    // assumes glViewport is set to 0, 0, wind. width, wind. height

    size_t width = feconfig.window_width;
    size_t height = feconfig.window_height;
    float fwid = (float)width;
    float fhgt = (float)height;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho( 0, width, 0, height, -1, 1 );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // set up shader

    impl->shader.use_shader();

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, impl->color_texname );
    glUniform1iARB( impl->shader.get_uniform_location("color_tex"), 0 );

    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, impl->depth_texname );
    glUniform1iARB( impl->shader.get_uniform_location("depth_tex"), 1 );
    
    glActiveTexture(GL_TEXTURE0);

    const FrameInfo* frame = impl->info;
    for ( size_t i = 0; i < frame->objects.size(); i++ ) {
        if ( frame->objects[i].rdata.type != common::LIGHTFIELD
          || frame->objects[i].rdata.pass_num != pass ) {
            continue;
        }

        glUniform1iARB( impl->shader.get_uniform_location("index"), i );

        float distance = length( frame->camera.get_position() - frame->objects[i].transform.position ); 
        glUniform1fARB( impl->shader.get_uniform_location("dist"), distance );

        const ClipRect& clip = frame->objects[i].clip;

        float cx = (float)clip.x;
        float cy = (float)clip.y;
        float cw = (float)clip.width;
        float ch = (float)clip.height;
        float maxs = cw / fwid;
        float maxt = ch / fhgt;

        glBegin( GL_QUADS);
        glTexCoord2f( 0.0f, 0.0f ); glVertex3f( cx,    cy,    0.0f );
        glTexCoord2f( maxs, 0.0f ); glVertex3f( cx+cw, cy,    0.0f );
        glTexCoord2f( maxs, maxt ); glVertex3f( cx+cw, cy+ch, 0.0f );
        glTexCoord2f( 0.0f, maxt ); glVertex3f( cx,    cy+ch, 0.0f );
        glEnd();
    }

    impl->shader.unuse_shader();
}

} // render
} /* frontend */
} /* aed */

