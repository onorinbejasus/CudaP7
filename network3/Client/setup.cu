// 
//  setup.cu
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-04-10.
//  Modified by Mitch Luban on 2011-04-12
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "imageio.hh"
#include "networking.hh"

#include "setup.hh"

#include <cutil.h>
#include <cuda_gl_interop.h>

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <string.h>
#include <stdarg.h>
#include <vector>

#define BLACK   0
#define RED     1
#define YELLOW  2
#define MAGENTA 3
#define GREEN   4
#define CYAN    5
#define BLUE    6
#define GREY    7
#define WHITE   8

#define MAXSAMPLES 100

extern int numCloths;
extern int sock;

int row	  = 40;
int column = 40;
unsigned int numTriangles = (row-1)*(column-1)*2;

GLuint vbo;
GLuint indexVbo;
unsigned int *flagIndexArray;
float *data_pointer;

int size = row * column;

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/
void free_data ( void )
{
    glDeleteBuffers(1, &indexVbo);
    glDeleteBuffers(1, &vbo);
	
	free(data_pointer);
    free(flagIndexArray);
}

/*--------------------------------------------------------------------
					Initialize System
--------------------------------------------------------------------*/

void init_system(void)
{		
    /******************************
     * Flag texturing and meshing
     * ***************************/

    flagIndexArray = (uint*)malloc(sizeof(uint) * numTriangles * 3);
	readline(sock, (uint*)flagIndexArray, sizeof(uint) * numTriangles * 3);

    glGenBuffers(1, &indexVbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * numTriangles * 3, flagIndexArray, GL_STATIC_DRAW);
	
	for(int ii = 0; ii < numTriangles * 3; ii++)
	{
		printf("index: %u\n", flagIndexArray[ii]);
	}
	
	// data_pointer initialize
	
	data_pointer = (float*)malloc(sizeof(float) * size * 3);
	writeline(sock, &vbo, sizeof(GLuint));
	readline(sock, (float*)data_pointer, sizeof(float) * size * 3);
	
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size * 3, data_pointer, GL_DYNAMIC_DRAW);
	
	for(int ii = 0; ii < size * 3; ii++)
	{
	 	printf("data: %f\n", data_pointer[ii]);
	}
}


/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

void draw_particles ( void )
{
	// Read from the socket
	writeline(sock, &vbo, sizeof(GLuint));
	readline(sock, (float*)data_pointer, sizeof(float) * size * 3);
	
	// Map to VBO
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size * 3, data_pointer, GL_DYNAMIC_DRAW);
	
	glPushMatrix();
		glColor3f(1.0, 1.0, 1.0);
		glTranslatef(0.0, 0.0, 0.0);
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
		glEnableClientState(GL_VERTEX_ARRAY);
		
		glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, 0);
		
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		glDisableClientState(GL_VERTEX_ARRAY);
	glPopMatrix(); 
}
