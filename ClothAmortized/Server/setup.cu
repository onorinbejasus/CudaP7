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

extern int numCloths;

const int threadsPerBlock = 256;
extern void verlet_simulation_step(struct Particle* pVector, float *data_pointer, float *norms, bool wind, int row, int column);

int row	  = 40;
int column = 40;
unsigned int numTriangles = (row-1)*(column-1)*2;

int width = 8;
int height = 4;

int texWidth;
int texHeight;

struct Particle* pVector;

unsigned int *flagIndexArray;
float *data_pointer;
float *gpuData_pointer;
float *flagNormals;
float *gpuFlagNormals;

float *flagTexArray;
GLuint flagTexId;
unsigned char *data;

int size = row * column;

extern bool dsim;
extern bool wind;

__device__
int getParticleInd(int x, int y, int row){
	return x + y * row;
}

float *get_dataPtr(){
	return data_pointer;
}

uint *get_indexPtr()
{
	return flagIndexArray;
}

float *get_flagTexArray() {
	return flagTexArray;
}

unsigned char *get_flagTexData() {
	return data;
}

float *get_flagNormals()
{
    return flagNormals;
}

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/
void free_data ( void )
{
	cudaFree(pVector);
	
	free(data_pointer);
	free(data);
    free(flagIndexArray);
    free(flagTexArray);
    free(flagNormals);
}

/*--------------------------------------------------------------------
					Make Particles
--------------------------------------------------------------------*/
__global__
void make_particles(struct Particle *pVector, float *data_pointer, float *norms, int row, int column, int width, int height)
{
	// //calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// 
	int i = index%row;
	int j = index/column;
	
	float3 pos = make_float3(width * (i/(float)row), -height * (j/(float)column), 0);
	
	if((j == 0 && i == 0) || (i == 0 && j == column-1))
		pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, norms, getParticleInd(i,j, row), false);
	else
		pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, norms, getParticleInd(i,j, row), true);
	
} // end make particles

/*--------------------------------------------------------------------
					Make Flag Mesh out of particles
--------------------------------------------------------------------*/

void make_flag_mesh( void )
{
    unsigned int currIndex = 0;

    float colFloat = (float)(column-1);
    float rowFloat = (float)(row-1);

    for(unsigned int ii = 0; ii < (size - column); ii++)
    {
        if( (ii+1) % column == 0 )
            continue;

        flagIndexArray[currIndex + 0] = ii + 0;
        flagIndexArray[currIndex + 1] = ii + column;
        flagIndexArray[currIndex + 2] = ii + 1;

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

    for(unsigned int ii = 0; ii < size; ii++)
    {
        int currX = column - ii%column;
        int currY = (ii/column)%row;
        flagTexArray[ii * 2 + 0] = (float)(currX)/colFloat;
		flagTexArray[ii * 2 + 1] = (float)(currY)/rowFloat;
    }
}

/*--------------------------------------------------------------------
					Initialize System
--------------------------------------------------------------------*/

void init_system(void)
{		
    data_pointer = (float*)malloc(sizeof(float) * size * 3);
	cudaMalloc((void**)&gpuData_pointer, sizeof(float) * size * 3);
	
    flagNormals = (float*)malloc(sizeof(float) * size * 3);
    cudaMalloc((void**)&gpuFlagNormals, sizeof(float) * size * 3);

	cudaMalloc( (void**)&(pVector), size * sizeof(struct Particle) );
	   
    /* create and copy */
    int totalThreads = row * column;
    int nBlocks = totalThreads/threadsPerBlock;
    nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;

    make_particles<<<nBlocks, threadsPerBlock>>>(pVector, gpuData_pointer, gpuFlagNormals, row, column, width, height); // create particles

	cudaMemcpy(data_pointer, gpuData_pointer, sizeof(float) * size * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(flagNormals, gpuFlagNormals, sizeof(float) * size * 3, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
	
    /******************************
     * Flag texturing and meshing
     * ***************************/

    flagIndexArray = (unsigned int*)malloc(sizeof(unsigned int) * numTriangles * 3);
    flagTexArray = (float*)malloc(sizeof(float) * size * 2);
    make_flag_mesh();

    const char *flagTextureFilename = "Textures/american_flag.png";
    data = loadImageRGBA(flagTextureFilename, &texWidth, &texHeight);
}
/*----------------------------------------------------------------------
relates mouse movements to tinker toy construction
----------------------------------------------------------------------*/
__global__
void remap_GUI(struct Particle *pVector, float *data_pointer, float *norms)
{	
	// //calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
		
	pVector[index].reset();
	data_pointer[index * 3 + 0] = pVector[index].m_ConstructPos.x;
	data_pointer[index * 3 + 1] = pVector[index].m_ConstructPos.y;
	data_pointer[index * 3 + 2] = pVector[index].m_ConstructPos.z;

    norms[index * 3 + 0] = 0.0f;
    norms[index * 3 + 1] = 0.0f;
    norms[index * 3 + 2] = 0.0f;
}

void step_func ( )
{
 	if ( dsim ){ // simulate
	    verlet_simulation_step(pVector, gpuData_pointer, gpuFlagNormals, wind, row, column);
		cudaMemcpy(data_pointer, gpuData_pointer, sizeof(float) * size * 3, cudaMemcpyDeviceToHost);
		cudaMemcpy(flagNormals, gpuFlagNormals, sizeof(float) * size * 3, cudaMemcpyDeviceToHost);
	}
    else { // remap

	    int totalThreads = row * column;
	    int nBlocks = totalThreads/threadsPerBlock;
	    nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
	
	    remap_GUI<<<nBlocks, threadsPerBlock>>>(pVector, gpuData_pointer, gpuFlagNormals);
		cudaMemcpy(data_pointer, gpuData_pointer, sizeof(float) * size * 3, cudaMemcpyDeviceToHost);
		cudaMemcpy(flagNormals, gpuFlagNormals, sizeof(float) * size * 3, cudaMemcpyDeviceToHost);
		
         cudaThreadSynchronize();
    }
    
}
