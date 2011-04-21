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

GLuint flagTexId;

GLuint vbo;
GLuint indexVbo;
GLuint texVbo;

unsigned int *flagIndexArray;
float *data_pointer;
float *flagTexArray;

int size = row * column;

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/
void free_data ( void )
{
    glDeleteBuffers(1, &indexVbo);
    glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &texVbo);
	
	free(data_pointer);
    free(flagIndexArray);
	free(flagTexArray);
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
	
	// Actual texture data 
	const char *flagTextureFilename = "Textures/american_flag.png";
    int w, h;
    unsigned char *data = loadImageRGBA(flagTextureFilename, &w, &h);
	
	glGenTextures(1, &flagTexId);
	glActiveTexture(GL_TEXTURE0_ARB);
	glBindTexture(GL_TEXTURE_2D, flagTexId);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	
	free(data);
	
	// Texture coordinates
	flagTexArray = (float*)malloc(sizeof(float) * size * 2);
	writeline(sock, &texVbo, sizeof(GLuint));
	readline(sock, (float*)flagTexArray, sizeof(float) * size * 2);
	
	glGenBuffers(1, &texVbo);
	glBindBuffer(GL_ARRAY_BUFFER, texVbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size * 2, flagTexArray, GL_STATIC_DRAW);
	
	// data_pointer initialize
	data_pointer = (float*)malloc(sizeof(float) * size * 3);
	glGenBuffers(1, &vbo);
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
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, flagTexId);
	
	glPushMatrix();
		glColor3f(1.0, 1.0, 1.0);
		glTranslatef(0.0, 0.0, 0.0);
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, texVbo);
		glTexCoordPointer(2, GL_FLOAT, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		
		glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, 0);
		
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glPopMatrix(); 
	
	glDisable(GL_TEXTURE_2D);
}
