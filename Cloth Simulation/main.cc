#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// open gl
#include "open_gl.hh"

#include <cuda_runtime.h>

#include "window.hh"
#include "setup.hh"
#include <iostream>
#include <cmath>

static bool right_click = false;
static bool left_click = false;

static int x_camera = 0, y_camera = 0, z_camera = 10;
static int lookAtX = 0, lookAtY = 0, lookAtZ = -1;

int dsim;

/// The display callback.
void display()
{
	glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT );
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity() ;
	gluLookAt(x_camera, y_camera, z_camera,  // x,y,z coord of the camera 
			  lookAtX, lookAtY, lookAtZ,
			  0,1,0); // the direction of Up (default is y-axis)
	
	step_func();
	
//	calculate_normals();
		
	draw_forces();
	draw_particles();

	glutSwapBuffers ();
	
	glutPostRedisplay();
}

void reshape(int width, int height) {
	glViewport(0, 0, width, height);
	glutPostRedisplay();
}

/// The keyboard callback
void keyboard(
	unsigned char key,	///< the key being pressed
	int x,				///< x coordinate of the mouse
	int y)				///< y coordinate of the mouse
{
	switch (key)
	{
		case 'c':	
		case 'C':
			clear_data ();
			break;
		case ' ':
			dsim = !dsim;
			break;
		
		case 'q':
		case 'Q':
		case 27:
			free_data ();
			cudaThreadExit();
			exit(0);
			break;
	
		default:
			break;
	}

	glutPostRedisplay();

}

 /// timer callback function
void my_timer(int id){

}

/// The mouse callback
void mouse(
	int button, ///< which button was pressesd
	int state,	///< up or down
	int x,		///< x position
	int y)		///< y position
{	
	return;
}

/// Mouse motion callback
void motion(
	int x,		///< x coordinate of mouse
	int y)		///< y coordinate of mouse
{
	return;
}

int main(int argc, char **argv) {
	createWindow(argc, argv);
	startApplication(argc, argv);
	return 0;
}


