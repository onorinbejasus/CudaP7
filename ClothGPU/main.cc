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

#define PI 3.14159265
#define movingSpeed 10.0

int numCloths;

static bool right_click = false;
static bool left_click = false;

// Camera variables
GLfloat posX, posY, posZ;
GLfloat cameraViewAngle, cameraSight;
GLfloat cameraSensitivity;

bool dsim = false;
bool wind = false;

/// The display callback.
void display()
{
    // Set the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 0.1, 100.0);

    // Set the modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Clear frame
    glClearColor(0.0f, 0.0f, 0.5f, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Set the camera
    gluLookAt(posX,
              posY,
              posZ,
              posX + cos(cameraViewAngle) * cos(cameraSight),
              posY + sin(cameraSight),
              posZ + sin(cameraViewAngle) * cos(cameraSight),
              cos(cameraViewAngle) * (-sin(cameraSight)),
              cos(cameraSight),
              sin(cameraViewAngle) * (-sin(cameraSight)));

	// step forward in time
	step_func();

    glPushMatrix();
    glTranslatef(-2.0f, 3.0f, -13.0f);
	draw_particles();
    glPopMatrix();

	// swap the back buffer
	glutSwapBuffers ();

	// recall display
	glutPostRedisplay();
}

void reshape(int width, int height) {
	glViewport(0, 0, width, height);

    // set the view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 0.1, 100.0);

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
        case 'j':
        case 'J':
            cameraViewAngle -= PI/cameraSensitivity;
            break;
        case 'l':
        case 'L':
            cameraViewAngle += PI/cameraSensitivity;
            break;
        case 'i':
        case 'I':
            cameraSight += 0.05;
            break;
        case 'k':
        case 'K':
            cameraSight -= 0.05;
            break;
        case 'a':
        case 'A':
            posX += movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            posZ -= movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            break;
        case 'd':
        case 'D':
            posX -= movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            posZ += movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            break;
        case 'w':
        case 'W':
            posX += movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            posZ += movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            break;
        case 's':
        case 'S':
            posX -= movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            posZ -= movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            break;
		case ' ':
			dsim = !dsim;
			break;
		case 'z':
		case 'Z':
			wind = !wind;
			break;
        case 'p':
        case 'P':
            printf("Camera: (%f, %f, %f,) at (%f, %f)\n", posX, posY, posZ, cameraViewAngle, cameraSight);
            break;
        case 'e':
        case 'E':
            posY -= 4.0/cameraSensitivity;
            break;
		case 'q':
		case 'Q':
            posY += 4.0/cameraSensitivity;
            break;
		case 27:
			free_data();
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

    // Init camera variables
    posX = 0.0; posY = 0.0; posZ = 0.0;
    cameraViewAngle = -1.5;
    cameraSight = 0.0;
    cameraSensitivity = 40.0;

    if(argc <= 1)
    {
        numCloths = 1;
    }
    else if(argc == 2)
    {
        numCloths = (int)atoi(argv[1]);
    }
    else
    {
        printf("Usage: ./main [numCloths]\n");
        return false;
    }

	createWindow(argc, argv);
	startApplication(argc, argv);
	return 0;
}


