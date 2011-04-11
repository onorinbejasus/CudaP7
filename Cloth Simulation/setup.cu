// 
//  setup.cu
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-04-10.
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

int width = 8;
int height = 8;

struct Particle* pVector;

GLuint vbo;
float4 *data_pointer;

int size = row * column;

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
void make_particles(struct Particle *pVector, float4 *data_pointer, int row, int column, int width, int height)
{
	// //calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// 
	int i = index%row;
	int j = index/column;
	
	float3 pos = make_float3(width * (i/(float)row), -height * (j/(float)column), 0);
	
	//if((j == 0 && i == 0) || (i == 0 && j == column-1))
	if(j == 0)	
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
	
	const int threadsPerBlock = 512;
	int totalThreads = row * column;
	int nBlocks = totalThreads/threadsPerBlock;
	nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
	
	make_particles<<<nBlocks, threadsPerBlock>>>(pVector, data_pointer, row, column, width, height); // create particles
		
	/* unmap vbo */
	cudaGLUnmapBufferObject(vbo);
	
}

/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

void draw_particles ( void )
{
	// render from the vbo
	glEnable(GL_POINT_SPRITE_ARB);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(shader);
    glUniform1f( glGetUniformLocation(shader, "pointScale"), 512 / tanf(150*0.5f*(float)M_PI/180.0f) );
    glUniform1f( glGetUniformLocation(shader, "pointRadius"), 0.02 );

    glColor3f(0, 1, 0);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

	glDrawArrays(GL_POINTS, 0, size);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glDisableClientState(GL_VERTEX_ARRAY); 
    glDisableClientState(GL_COLOR_ARRAY);

    glUseProgram(0);
    glDisable(GL_POINT_SPRITE_ARB);
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
		
		const int threadsPerBlock = 512;
		int totalThreads = row * column;
		int nBlocks = totalThreads/threadsPerBlock;
		nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
		
		remap_GUI<<<nBlocks, threadsPerBlock>>>(pVector, data_pointer);
		
		/* unmap vbo */
		cudaGLUnmapBufferObject(vbo);
		
	}
}