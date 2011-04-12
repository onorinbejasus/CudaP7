/**
 * @file material.cpp
 * @brief Material class
 *
 * @author Eric Butler (edbutler)
 */

#include "material.hpp"
#include "common/util/imageio.hpp"

namespace aed {
namespace frontend {
namespace render {

Material::Material():
    ambient( Color3::White ),
    diffuse( Color3::White ),
    specular( Color3::Black ),
    shininess( 10.0 ),
    tex_width( 0 ),
    tex_height( 0 ),
    tex_data( 0 )
{
    tex_handle = 0;
}

Material::~Material()
{
    if ( tex_data ) {
        free( tex_data );
        if ( tex_handle ) {
            glDeleteTextures( 1, &tex_handle );
        }
    }
}

bool Material::load()
{
    // if data has already been loaded, clear old data
    if ( tex_data ) {
        free( tex_data );
        tex_data = 0;
    }

    // if no texture, nothing to do
    if ( texture_filename.empty() )
        return false;

    std::cout << "Loading texture " << texture_filename << "...\n";


    tex_data = common::util::imageio_load_image( texture_filename.c_str(), &tex_width, &tex_height );
    if ( !tex_data ) {
        std::cerr << "Cannot load texture file " << texture_filename << std::endl;
        return false;
    }

    std::cout << "Finished loading texture" << std::endl;
    return true;
}

bool Material::create_gl_data()
{
    // if no texture, nothing to do
    if ( texture_filename.empty() )
        return false;

    if ( !tex_data ) {
        return false;
    }

    // clean up old texture
    if ( tex_handle ) {
        glDeleteTextures( 1, &tex_handle );
    }

    assert( tex_width > 0 && tex_height > 0 );

    glGenTextures( 1, &tex_handle );
    if ( !tex_handle ) {
        return false;
    }

    glBindTexture( GL_TEXTURE_2D, tex_handle );
    //glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data );
    gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGBA, tex_width, tex_height, GL_RGBA, GL_UNSIGNED_BYTE, tex_data );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );

    glBindTexture( GL_TEXTURE_2D, 0 );

    std::cout << "Loaded GL texture " << texture_filename << '\n';
    return true;
}

void Material::set_gl_state() const
{
    float arr[4];
    arr[3] = 1.0; // alpha always 1.0
	glEnable(GL_TEXTURE_2D);
    // always bind, because if no texture this will set texture to nothing
    glBindTexture( GL_TEXTURE_2D, tex_handle );

    ambient.to_array( arr );
    glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT,   arr );
    diffuse.to_array( arr );
    glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE,   arr );
    specular.to_array( arr );
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR,  arr );
    // make up a shininess term
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, shininess );
}

void Material::reset_gl_state() const
{
    glBindTexture( GL_TEXTURE_2D, 0 );
}

} // render
} // frontend
} /* aed */

