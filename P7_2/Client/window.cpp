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

// callbacks
extern void display();
extern void keyboard(unsigned char key, int x, int y);
extern void mouse(int button, int state, int x, int y);
extern void motion(int x, int y);

// GLUT specific variables
unsigned int window_width = 512;
unsigned int window_height = 512;

unsigned int timer = 0; // a timer for FPS calculations

// Forward declarations of GL functionality
CUTBoolean initGL(int argc, char** argv);

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
		sprintf(fps, "15-668 - Project 1: %3.1f fps ", ifps);	
		
		glutSetWindowTitle(fps);
		fpsCount = 0; 
		
		cutilCheckError(cutResetTimer(timer));	
	}
}

void fpsDisplay()
{
	cutilCheckError(cutStartTimer(timer));	
	
	display();
	
	cutilCheckError(cutStopTimer(timer));
	computeFPS();
}

CUTBoolean createWindow(int argc, char ** argv)
{
	// Create the CUTIL timer
	cutilCheckError( cutCreateTimer( &timer));
	
	if (CUTFalse == initGL(argc, argv)) {
		return CUTFalse;
	}

	initCuda(argc, argv);
	CUT_CHECK_ERROR_GL();

	// register callbacks
	glutDisplayFunc(fpsDisplay);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
}

void startApplication(int argc, char ** argv)
{
	// start rendering mainloop
	glutMainLoop();
	
	// clean up
	cudaThreadExit();
	cutilExit(argc, argv);
}

CUTBoolean initGL(int argc, char **argv)
{
	//Steps 1-2: create a window and GL context (also register callbacks)
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("15-668 - Project 7");
	glutDisplayFunc(fpsDisplay);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	
	// check for necessary OpenGL extensions
	glewInit();
	if (! glewIsSupported( "GL_VERSION_2_0 " ) ) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		return CUTFalse;
	}
	
	// Step 3: Setup our viewport and viewing modes
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );

	glEnable(GL_DEPTH_TEST);
	glClearColor(0,0,0,1);
	glMatrixMode(GL_PROJECTION) ;
	glLoadIdentity() ;
	gluPerspective(20, 1.0, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity() ;  // init modelview to identity
	
	// init_system();
	
	return CUTTrue;
}
