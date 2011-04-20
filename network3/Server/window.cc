// simpleGLmain.cpp (Rob Farber)
// http://www.drdobbs.com/architecture-and-design/222600097

// open gl
#include "open_gl.hh"

// cuda utility libraries
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <cuda.h>

#include "setup.hh"

unsigned int timer = 0; // a timer for FPS calculations

// First initialize OpenGL context, so we can properly set the GL
// for CUDA. NVIDIA notes this is necessary in order to achieve
// optimal performance with OpenGL/CUDA interop.	use command-line
// specified CUDA device, otherwise use device with highest Gflops/s
void initCuda(int argc, char** argv)
{
	
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
		cutilGLDeviceInit(argc, argv);
	} else {
		cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
	}

	// Clean up on program exit
//	atexit(free_data);
}

// Simple method to display the Frames Per Second in the window title
void computeFPS()
{
	  int fpsCount=0;
	  int fpsLimit=100;

	fpsCount++;
	
	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
		sprintf(fps, "GPU CLOTH", ifps);	
		
		glutSetWindowTitle(fps);
		fpsCount = 0; 
		
		cutilCheckError(cutResetTimer(timer));	
	}
}

bool initGL(int argc, char **argv)
{
	//Steps 1-2: create a window and GL context (also register callbacks)
//	glutInit(&argc, argv);
//	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	// 
	// // check for necessary OpenGL extensions
	// glewInit();
	// if (! glewIsSupported( "GL_VERSION_2_0 " ) ) {
	// 	fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
	// 	return false;
	// }
	// 
	init_system();
	
	return true;
}