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
#include <iostream>
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

// Lighting attributes
GLfloat lightPos[] = {3.0, 5.0, -4.0, 0.0};
GLfloat lightColor[] = {1.0, 1.0, 1.0, 1.0};
GLfloat lightSpecular[] = {1.0, 1.0, 1.0, 1.0};
GLfloat lightDiffuse[] = {1.0, 1.0, 1.0, 1.0};
GLfloat lightShine[] = {20.0};

GLuint flagTexId;

GLuint vbo;
GLuint indexVbo;
GLuint texVbo;
GLuint normVbo;

unsigned int *flagIndexArray;
float *data_pointer;
float *flagTexArray;
float *flagNormArray;

int size = row * column;

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/
void free_data ( void )
{
    glDeleteBuffers(1, &indexVbo);
    glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &texVbo);
    glDeleteBuffers(1, &normVbo);
	
	free(data_pointer);
    free(flagIndexArray);
	free(flagTexArray);
    free(flagNormArray);
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

	printf("numTriangles %u\n", numTriangles);

	for(int ii = 0; ii < numTriangles * 3; ii++)
	{
		printf("index: %u\n", flagIndexArray[ii]);
	}

    glGenBuffers(1, &indexVbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * numTriangles * 3, flagIndexArray, GL_STATIC_DRAW);
	
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

    // flagNormals initialize
    flagNormArray = (float*)malloc(sizeof(float) * size * 3);
    glGenBuffers(1, &normVbo);

    /*******************
     * Lighting stuff
     * ****************/

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);

    // Enable lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Set the light color and position
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_POSITION, lightColor);
}


/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

void draw_particles ( void )
{
    // Enable lighting
    glEnable(GL_LIGHTING);

	// Read from the socket
	memset(data_pointer, 0 ,sizeof(float) * size * 3);	
	
	int n = 0;

	writeline(sock, &vbo, sizeof(GLuint));

	n = readline(sock, (float*)data_pointer, sizeof(float) * size * 3);

	while(n < (sizeof(float) * size * 3)){
		n += readline(sock, (char*)data_pointer + n, (sizeof(float) * size * 3) - n);

	}	

    memset(flagNormArray, 0, sizeof(float) * size * 3);

    n = 0;

    writeline(sock, &normVbo, sizeof(GLuint));

    n = readline(sock, (float*)flagNormArray, sizeof(float) * size * 3);

    while(n < (sizeof(float) * size * 3))
    {
        n += readline(sock, (char*)flagNormArray + n, (sizeof(float) * size * 3) - n);
    }

	// Map to VBO
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size * 3, data_pointer, GL_DYNAMIC_DRAW);

    // Map to normal VBO
    glBindBuffer(GL_ARRAY_BUFFER, normVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size * 3, flagNormArray, GL_DYNAMIC_DRAW);
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, flagTexId);

    glMaterialfv(GL_FRONT, GL_SPECULAR, lightSpecular);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, lightDiffuse);
    glMaterialfv(GL_FRONT, GL_SHININESS, lightShine);
	
	glPushMatrix();
		glColor3f(1.0, 1.0, 1.0);
		glTranslatef(0.0, 0.0, 0.0);
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, texVbo);
		glTexCoordPointer(2, GL_FLOAT, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, normVbo);
        glNormalPointer(GL_FLOAT, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
		glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		
		glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, 0);
		
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		
		glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glPopMatrix(); 
	
	glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
}
