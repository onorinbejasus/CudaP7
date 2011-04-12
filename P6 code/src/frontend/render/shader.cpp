
#include "shader.hpp"
#include "common/log.hpp"
#include "frontend/config.hpp"
#include <fstream>
#include <iostream>
#include <stdlib.h>

namespace aed {
namespace frontend {
namespace render {

using namespace common;
using namespace util;

///////////////////////////////////////////////////////////////////////////////
// functions for common GLSL usage

//Got this from http://www.lighthouse3d.com/opengl/glsl/index.php?oglinfo
// it prints out shader info (debugging!)
void printShaderInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("printShaderInfoLog: %s\n",infoLog);
        free(infoLog);
	}else{
		printf("Shader Info Log: OK\n");
	}
}

//Got this from http://www.lighthouse3d.com/opengl/glsl/index.php?oglinfo
// it prints out shader info (debugging!)
void printProgramInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;
	glGetProgramiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("printProgramInfoLog: %s\n",infoLog);
        free(infoLog);
    }else{
		printf("Program Info Log: OK\n");
	}
}


char* read_file(const char* filename)
{
    std::ifstream infile( filename );
    if( infile.fail() ) {
        LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "ERROR: cannot open file:%s.", filename);
        infile.close();
        return 0;
    }

    // calculate length
    infile.seekg( 0, std::ios::end );
    int length = infile.tellg();
    infile.seekg( 0, std::ios::beg );
    // allocate space for entire program
    char* const buffer = (char *) malloc( (length + 1) * sizeof *buffer );
    if ( !buffer ) {
        LOG_MSG(MDL_FRONT_END, SVR_INIT, "ERROR: cannot allocate memory for loading shader.");
        return 0;
	}
    // copy entire program into memory
    infile.getline( buffer, length, '\0' );
    infile.close();

    return buffer;
}

bool load_shader( const char* file, GLint type, GLhandleARB program ) {
    assert( file );

    GLhandleARB shader;
    char error_msg[16384];
    error_msg[0] = '\0';

    char* buffer = read_file(file);
    if (!buffer)
        return false;

    // create shader object
    shader = glCreateShader( type );
    // link shader source
    const char* src = buffer; // workaround for const correctness
    glShaderSource( shader, 1, &src, NULL );
    // compile shaders
    glCompileShader( shader );

    // check success
    GLint result;
    glGetObjectParameterivARB( shader, GL_OBJECT_COMPILE_STATUS_ARB, &result );
    if ( result != GL_TRUE ) {
        glGetInfoLogARB( shader, sizeof error_msg, NULL, error_msg );
        LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "GLSL COMPILE ERROR in %s : %s. ", file, error_msg);
        return false;
    } 
	else {
        LOG_MSG(MDL_FRONT_END, SVR_INIT, "Compiled shaders successfully");
    }

    // attach the shader object to the program object
    glAttachShader( program, shader );
    
	printShaderInfoLog(shader);

    free( buffer );
    return true;
}

void check_gl_state(const char* filename, const int linenumber)
{
    GLenum gl_status = glGetError();
    if (GL_NO_ERROR != gl_status)  {
        std::cout << filename << ":" << linenumber << " Found OpenGL error: " 
                  << gluErrorString(gl_status) << std::endl;
        assert(false);
    }
}

// XXX what exactly does the return value mean?

bool check_program_object(GLuint program)
{
    if (!glIsProgram(program)) {
        printf("current program is corrupted.\n");
        assert(false);
    }
	return true;
}

///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// ShaderProgram class
//


void ShaderProgram::destroy_shaders() 
{
    // unbind the shader
    glUseProgramObjectARB(0);
    if (GL_TRUE == glIsProgram(_program)) {
        glDeleteObjectARB( _program );
        _program = 0;
    }
}

GLint ShaderProgram::get_uniform_location(const char* uniform_name)
{
    check_program_object(_program);
    return glGetUniformLocationARB(_program, uniform_name);
}

// XXX why do these return booleans?

bool ShaderProgram::use_shader()
{
    check_program_object(_program);
    glUseProgramObjectARB(_program);
	return true;
}

bool ShaderProgram::unuse_shader()
{
    check_program_object(_program);
    glUseProgramObjectARB( 0 );
	return true;
}

bool ShaderProgram::init_shader( 
    const char* vert_file, const char* geom_file, const char* frag_file, 
    GLenum geomInputType, GLenum geomOutputType )
{
    bool rv = true;

    // OpenGL crashes if I directly assign to member variable, don't know why
    GLuint _program = glCreateProgram();
    LOG_VAR_MSG( MDL_FRONT_END, SVR_INIT, "The Shader is %d", _program);

    LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "Loading vertex shader  :%s.", vert_file);
    LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "Loading geometry shader:%s.", geom_file);
    LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "Loading fragment shader:%s.", frag_file);

    rv = rv && load_shader( vert_file, GL_VERTEX_SHADER_ARB,   _program );  // Load vertex shader
    rv = rv && load_shader( geom_file, GL_GEOMETRY_SHADER_ARB, _program );  // Load geomethy shader
    rv = rv && load_shader( frag_file, GL_FRAGMENT_SHADER_ARB, _program );  // Load fragment shader

    glProgramParameteriARB(_program, GL_GEOMETRY_INPUT_TYPE_EXT, geomInputType);
    glProgramParameteriARB(_program, GL_GEOMETRY_OUTPUT_TYPE_EXT, geomOutputType);
    int temp;
    glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &temp);
    glProgramParameteriARB(_program, GL_GEOMETRY_VERTICES_OUT_EXT, temp);

    if ( !rv ) 
        return false;
    

    // link
	printProgramInfoLog(_program);
    glLinkProgram( _program );
	printProgramInfoLog(_program);
    this->_program = _program;


    // check for success
    GLint result;
    glGetProgramiv( _program, GL_LINK_STATUS, &result );
    if ( result == GL_TRUE ) {
       LOG_MSG(MDL_FRONT_END, SVR_INIT, "Successfully linked shader");
       return true;
    } else {
       LOG_MSG(MDL_FRONT_END, SVR_INIT, "FAILED to link shader");
       return false;
    }
}

bool ShaderProgram::init_shader( const char* vert_file, const char* frag_file )
{
    // OpenGL crashes if I directly assign to member variable, don't know why
    GLuint _program = glCreateProgram();
    LOG_VAR_MSG( MDL_FRONT_END, SVR_INIT, "The Shader is %d", _program);

    bool rv = true;
    LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "Loading vertex shader  :%s.", vert_file);
    LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "Loading fragment shader:%s.", frag_file);

    rv = rv && load_shader( vert_file, GL_VERTEX_SHADER_ARB,   _program );  // Load vertex shader
    rv = rv && load_shader( frag_file, GL_FRAGMENT_SHADER_ARB, _program );  // Load fragment shader

    if ( !rv ) 
        return false;
	
    // link
    glLinkProgramARB( _program );
	printProgramInfoLog(_program);
    
    this->_program = _program;

    // check for success
    GLint result;
    glGetProgramiv( _program, GL_LINK_STATUS, &result );
    if ( result == GL_TRUE ) {
       LOG_MSG(MDL_FRONT_END, SVR_INIT, "Successfully linked shader");
       return true;
    } else {
       LOG_MSG(MDL_FRONT_END, SVR_INIT, "FAILED to link shader");
       return false;
    }
}

} // render
} /* frontend */
} /* aed */

