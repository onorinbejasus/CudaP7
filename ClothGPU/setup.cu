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

static const int threadsPerBlock = 64;

int row	  = 40;
int column = 40;
unsigned int numTriangles = (row-1)*(column-1)*2;

int width = 8;
int height = 4;

struct Particle** pVector;

// Lighting attributes
GLfloat lightPos[] = {3.0, 5.0, -4.0, 0.0};
GLfloat lightColor[] = {1.0, 1.0, 1.0, 1.0};
GLfloat lightSpecular[] = {1.0, 1.0, 1.0, 1.0};
GLfloat lightDiffuse[] = {1.0, 1.0, 1.0, 1.0};
GLfloat lightShine[] = {20.0};

// Vbos
GLuint *vbo;
GLuint *normVbo;
GLuint texVbo;
GLuint indexVbo;

// Flag normals
float3 **flagNormals;

// Triangle indices
unsigned int *flagIndexArray;

// Flag positions
float4 **data_pointer;

// Flag texture coordinates
float2 *flagTexArray;

// Texture image id
GLuint flagTexId;

int size = row * column;

extern void verlet_simulation_step(struct Particle* pVector, float4 *data_pointer, GLuint vbo, float3 *flagNorms, GLuint nVbo, bool wind, int row, int column, int numCloth);
void deleteVBO(int numCloth);
void deleteTexVBO();
void deleteIndexVBO();
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
    for(int ii = 0; ii < numCloths; ii++)
    {
	    cudaFree(pVector[ii]);
	    deleteVBO(ii);

        free(flagNormals[ii]);
    }

    free(flagNormals);

    glDeleteBuffers(1, &indexVbo);
    glDeleteBuffers(1, &texVbo);

    free(flagIndexArray);
    free(flagTexArray); 
}

/*--------------------------------------------------------------------
					Make Particles
--------------------------------------------------------------------*/
__global__
void make_particles(struct Particle *pVector, float4 *data_pointer, float3 *flagNorms, int row, int column, int width, int height)
{
	// //calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// 
	int i = index%row;
	int j = index/column;
	
	float3 pos = make_float3(width * (i/(float)row), -height * (j/(float)column), 0);
	
	if((j == 0 && i == 0) || (i == 0 && j == column-1))
		pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, flagNorms, getParticleInd(i,j, row), false);
	else
		pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, flagNorms, getParticleInd(i,j, row), true);
	
} // end make particles

/*----------------------------------------------------------
					Create VBO
----------------------------------------------------------*/

void createVBO(int clothNum)
{
    // create buffer object
    glGenBuffers( 1, &(vbo[clothNum]));
    glBindBuffer( GL_ARRAY_BUFFER, vbo[clothNum]);

    // initialize buffer object
    unsigned int m_size = size * 4 * sizeof(float);
    glBufferData( GL_ARRAY_BUFFER, m_size, 0, GL_DYNAMIC_DRAW);

    // register buffer object with CUDA
    cudaGLRegisterBufferObject(vbo[clothNum]);
}

/*----------------------------------------------------------
					Delete VBO
----------------------------------------------------------*/
void deleteVBO(int clothNum)
{
    glBindBuffer( 1, vbo[clothNum]);
    glDeleteBuffers( 1, &(vbo[clothNum]));

    glBindBuffer( 1, normVbo[clothNum]);
    glDeleteBuffers( 1, &(normVbo[clothNum]));

    cudaGLUnregisterBufferObject(vbo[clothNum]);
    cudaGLUnregisterBufferObject(normVbo[clothNum]);
}

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
        flagTexArray[ii] = make_float2((float)currX/colFloat, (float)(currY)/rowFloat);
    }
}


/*--------------------------------------------------------------------
					Initialize System
--------------------------------------------------------------------*/

void init_system(void)
{		
    vbo = (GLuint*)malloc(sizeof(GLuint) * numCloths);
    normVbo = (GLuint*)malloc(sizeof(GLuint) * numCloths);
    pVector = (Particle**)malloc(sizeof(Particle*) * numCloths);
    data_pointer = (float4**)malloc(sizeof(float4*) * numCloths);
    flagNormals = (float3**)malloc(sizeof(float3*) * numCloths);

	/* malloc cuda memory*/
    for(int ii = 0; ii < numCloths; ii++)
    {
	    cudaMalloc( (void**)&(pVector[ii]), size * sizeof(struct Particle) );
		
	    /* initialize VBO */
        glGenBuffers(1, &(vbo[ii]));
        glBindBuffer(GL_ARRAY_BUFFER, vbo[ii]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * size, 0, GL_DYNAMIC_DRAW);
        cudaGLRegisterBufferObject(vbo[ii]);

        glGenBuffers(1, &(normVbo[ii]));
        glBindBuffer(GL_ARRAY_BUFFER, normVbo[ii]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * size, 0, GL_DYNAMIC_DRAW);
        cudaGLRegisterBufferObject(normVbo[ii]);

	    /* map vbo in cuda */
	    cudaGLMapBufferObject((void**)&(data_pointer[ii]), vbo[ii]);
	    cudaGLMapBufferObject((void**)&(flagNormals[ii]), normVbo[ii]);
	
	    /* create and copy */
	    int totalThreads = row * column;
	    int nBlocks = totalThreads/threadsPerBlock;
	    nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
	
	    make_particles<<<nBlocks, threadsPerBlock>>>(pVector[ii], data_pointer[ii], flagNormals[ii], row, column, width, height); // create particles

        cudaThreadSynchronize();
		
	    /* unmap vbo */
	    cudaGLUnmapBufferObject(vbo[ii]);
	    cudaGLUnmapBufferObject(normVbo[ii]);
    }

    /******************************
     * Flag texturing and meshing
     * ***************************/

    flagIndexArray = (unsigned int*)malloc(sizeof(unsigned int) * numTriangles * 3);
    flagTexArray = (float2*)malloc(sizeof(float2) * size);
    // flagTexArray = (float*)malloc(sizeof(float) * numTriangles * 3 * 2);
    make_flag_mesh();

    glGenBuffers(1, &indexVbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * numTriangles * 3, flagIndexArray, GL_STATIC_DRAW);

    glGenBuffers(1, &texVbo);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * size, flagTexArray, GL_STATIC_DRAW);

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

    /******************************
     * Lighting
     * ***************************/
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);

    // Enable lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Set the light position and color
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor);
}

/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

void draw_particles ( void )
{
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, flagTexId);

    glMaterialfv(GL_FRONT, GL_SPECULAR, lightSpecular);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, lightDiffuse);
    glMaterialfv(GL_FRONT, GL_SHININESS, lightShine);

	// iterate over the cloth particles and draw their corresponding mesh
    for(int ii = 0; ii < numCloths; ii++)
    {
        glPushMatrix();
	        glColor3f(1.0, 1.0, 1.0);
            glTranslatef(ii*10, 0.0, 0.0);

            glBindBuffer(GL_ARRAY_BUFFER, vbo[ii]);
            glVertexPointer(4, GL_FLOAT, 0, (GLvoid*)((char*)NULL));
            glBindBuffer(GL_ARRAY_BUFFER, texVbo);
            glTexCoordPointer(2, GL_FLOAT, 0, (GLvoid*)((char*)NULL));
            glBindBuffer(GL_ARRAY_BUFFER, normVbo[ii]);
            glNormalPointer(GL_FLOAT, 0, (GLvoid*)((char*)NULL));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_NORMAL_ARRAY);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            // Render flag mesh
            glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, (GLvoid*)((char*)NULL));

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            glDisableClientState(GL_VERTEX_ARRAY); 
            glDisableClientState(GL_NORMAL_ARRAY);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glPopMatrix(); 
    }

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
}

/*----------------------------------------------------------------------
relates mouse movements to tinker toy construction
----------------------------------------------------------------------*/
__global__
void remap_GUI(struct Particle *pVector, float4 *data_pointer, float3 *flagNorms)
{	
	//calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// reset particles	
	pVector[index].reset();

	// reset vbo and texture normals
	data_pointer[index] = make_float4(pVector[index].m_ConstructPos, 1);
    flagNorms[index] = make_float3(0.0f, 0.0f, -1.0f);
}

void step_func ( )
{
	// iterate of the number of cloth
	
    for(int ii = 0; ii < numCloths; ii++)
    {
	    if ( dsim ){ // simulate
		    verlet_simulation_step(pVector[ii], data_pointer[ii], vbo[ii], flagNormals[ii], normVbo[ii], wind, row, column, ii);
	    }
	    else { // remap
		    /* map vbo in cuda */
		    cudaGLMapBufferObject((void**)&(data_pointer[ii]), vbo[ii]);
		    cudaGLMapBufferObject((void**)&(flagNormals[ii]), normVbo[ii]);
		
		    int totalThreads = row * column;
		    int nBlocks = totalThreads/threadsPerBlock;
		    nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
		
			// launch kernel to remap
		    remap_GUI<<<nBlocks, threadsPerBlock>>>(pVector[ii], data_pointer[ii], flagNormals[ii]);

            cudaThreadSynchronize();
		
		    /* unmap vbo */
		    cudaGLUnmapBufferObject(vbo[ii]);
		    cudaGLUnmapBufferObject(normVbo[ii]);
	    }
    }
}
