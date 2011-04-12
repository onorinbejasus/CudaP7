// 
//  setup.cu
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-04-10.
//  Modified by Mitch Luban on 2011-04-12
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "imageio.hh"
#include "Particle.hh"

#include "setup.hh"

#include <cutil.h>
#include <cuda_gl_interop.h>

#include <stdlib.h>
#include <stdio.h>
#include <cmath>

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

int row	  = 40;
int column = 40;
unsigned int numTriangles = (row-1)*(column-1)*2;

int width = 8;
int height = 4;

struct Particle* pVector;

GLuint vbo;
unsigned int *flagIndexArray;
float4 *data_pointer;

float *flagTexArray;
GLuint flagTexId;

int size = row * column;

float2 offset;

extern void verlet_simulation_step(struct Particle* pVector, float4 *data_pointer, GLuint vbo, bool wind, int row, int column);
void deleteVBO(GLuint *vbo);
extern bool dsim;
extern bool wind;
extern GLuint shader;

__device__
int getParticleInd(int x, int y, int row) { return y*row+x; }

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/
void free_data ( void )
{
	cudaFree(pVector);
	deleteVBO(&vbo);
	data_pointer = 0;
}

/*--------------------------------------------------------------------
					Make Particles
--------------------------------------------------------------------*/
__global__
void make_particles(struct Particle *pVector, float4 *data_pointer, int row, int column, int width, int height, float2 offset)
{
	// //calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// 
	int i = index%row;
	int j = index/column;
	
	float3 pos = make_float3(offset.x + (width * (i/(float)row) ), offset.y + (-height * (j/(float)column) ), 0);
	
	if((j == 0 && i == 0) || (i == 0 && j == column-1))
		pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, getParticleInd(i,j, row), false);
	else
		pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, getParticleInd(i,j, row), true);
	
} // end make particles

/*----------------------------------------------------------
					Create VBO
----------------------------------------------------------*/

void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int m_size = size * 4 * sizeof( float);
    glBufferData( GL_ARRAY_BUFFER, m_size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    cudaGLRegisterBufferObject(*vbo);
}

/*----------------------------------------------------------
					Delete VBO
----------------------------------------------------------*/
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    cudaGLUnregisterBufferObject(*vbo);

    *vbo = 0;
}

/*--------------------------------------------------------------------
					Make Flag Mesh out of particles
--------------------------------------------------------------------*/

void make_flag_mesh( void )
{
    unsigned int currIndex = 0;

    for(unsigned int ii = 0; ii < (size - column); ii++)
    {
        if( (ii+1) % column == 0 )
            continue;

        flagIndexArray[currIndex + 0] = ii + 0;
        flagIndexArray[currIndex + 1] = ii + 1;
        flagIndexArray[currIndex + 2] = ii + column;

        currIndex += 3;
    }

    for(unsigned int ii = row; ii < size; ii++)
    {
        if( (ii+1) % column == 0 )
            continue;

        flagIndexArray[currIndex + 0] = ii + 0;
        flagIndexArray[currIndex + 1] = ii + 1;
        flagIndexArray[currIndex + 2] = (ii + 1) - column;

        currIndex += 3;
    }
}

/*--------------------------------------------------------------------
					Initialize System
--------------------------------------------------------------------*/

void init_system(void)
{		
	/* malloc cuda memory*/
	cudaMalloc( (void**)&pVector, size * sizeof(struct Particle) );
		
	/* initialize VBO */
	createVBO(&vbo);
	
	/* map vbo in cuda */
	cudaGLMapBufferObject((void**)&data_pointer, vbo);
	
	/* create and copy */
	
	const int threadsPerBlock = 64;
	int totalThreads = row * column;
	int nBlocks = totalThreads/threadsPerBlock;
	nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
	
	offset = make_float2(-10.0, 10.0f);
	
	make_particles<<<nBlocks, threadsPerBlock>>>(pVector, data_pointer, row, column, width, height, offset); // create particles
		
	/* unmap vbo */
	cudaGLUnmapBufferObject(vbo);

    /******************************
     * Flag texturing and meshing
     * ***************************/

    flagIndexArray = (unsigned int*)malloc(sizeof(unsigned int) * numTriangles * 3);
    make_flag_mesh();

    const char *flagTextureFilename = "Textures/american_flag.png";
    int w, h;
    unsigned char *data = loadImageRGBA(flagTextureFilename, &w, &h);

    glGenTextures(1, &flagTexId);
    glBindTexture(GL_TEXTURE_2D, flagTexId);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    free(data);
}


/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

void draw_particles ( void )
{
    glPushMatrix();
	    glColor3f(0, 1, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
        // glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        // Set up vertices
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);

        // glTexCoordPointer(2, GL_FLOAT, 0, flagTexArray);

        // glDrawElements(GL_POINTS, size, GL_UNSIGNED_INT, flagIndexArray);
        glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, flagIndexArray);

        // Why is this here?
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

        glDisableClientState(GL_VERTEX_ARRAY); 
        // glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glPopMatrix(); 
}

/*--------------------------------------------------------------------
					Draw Foreces
--------------------------------------------------------------------*/
void draw_forces ( void )
{}

/*----------------------------------------------------------------------
relates mouse movements to tinker toy construction
----------------------------------------------------------------------*/
__global__
void remap_GUI(struct Particle *pVector, float4 *data_pointer)
{	
	// //calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
		
	pVector[index].reset();
	data_pointer[index] = make_float4(pVector[index].m_ConstructPos, 1);
}

void step_func ( )
{
	if ( dsim ){ // simulate
		verlet_simulation_step(pVector, data_pointer, vbo, wind, row, column);
	}
	else { // remap
		
	// remove old 
		deleteVBO(&vbo);
		data_pointer = 0;

		/* initialize VBO */
		createVBO(&vbo);

		/* map vbo in cuda */
		cudaGLMapBufferObject((void**)&data_pointer, vbo);
		
		const int threadsPerBlock = 64;
		int totalThreads = row * column;
		int nBlocks = totalThreads/threadsPerBlock;
		nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
		
		remap_GUI<<<nBlocks, threadsPerBlock>>>(pVector, data_pointer);
		
		/* unmap vbo */
		cudaGLUnmapBufferObject(vbo);
		
	}
}
