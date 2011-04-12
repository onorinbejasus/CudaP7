/// \file System-dependent includes for openGL


#ifdef __APPLE__

#include <GL/glew.h> // must be included before OpenGL
#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#else

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>

#endif
