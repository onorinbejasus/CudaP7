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
#include "Constraint.hh"

#include "setup.hh"

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
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

int row	  = 60;
int column = 60;
unsigned int numTriangles = (row-1)*(column-1)*2;

int width = 8;
int height = 4;

struct Particle** pVector;
static std::vector<Constraint*> constraints;

GLuint *vbo;
GLuint texVbo;
GLuint indexVbo;
unsigned int *flagIndexArray;
float4 **data_pointer;

float2 *flagTexArray;
GLuint flagTexId;

int size = row * column;

extern void verlet_simulation_step(struct Particle* pVector, std::vector<Constraint*> constraints, 
												float4 *data_pointer, GLuint vbo, int row, int column, bool wind, int numCloths);
void deleteVBO(int numCloth);
void deleteTexVBO();
void deleteIndexVBO();

extern bool dsim;
extern bool wind;

int getParticleInd(int x, int y, int row) { return y*row+x; }
struct Particle *getParticle(int x, int y){ return &pVector[0][y*row+x]; }

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/
void free_data ( void )
{
    for(int ii = 0; ii < numCloths; ii++)
    {
        free(pVector[ii]);
	    deleteVBO(ii);
        free(data_pointer[ii]);
    }
    free(data_pointer);
    free(pVector);

    glDeleteBuffers(1, &indexVbo);
    glDeleteBuffers(1, &texVbo);

    free(flagIndexArray);
    free(flagTexArray);
}
/*--------------------------------------------------------------------
					Make Constraints
--------------------------------------------------------------------*/

void make_constraints(void){
	
	for(int ii = 0; ii < row; ii++){
		
		for(int jj = 0; jj < column; jj++){
						
			/* neighbors */
						
			if(ii < row-1) // to the right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+1, jj)) );
			
			if(jj < column -1) // below	
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii,jj+1) ) );
			
			if(ii < row-1 && jj < column -1) // down right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+1,jj+1) ) );
			
			if(ii < row-1 && jj < column -1) // up right	
				constraints.push_back( new Constraint(getParticle(ii+1,jj), getParticle(ii,jj+1) ) );
			
			/* neighbor's neighbors */
			
		 	if(ii < row-2) // to the right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+2, jj) ) );
			
			if(jj < column -2) // below	
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii,jj+2) ) );
			
			if(ii < row-2 && jj < column -2) // down right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+2,jj+2) ) );
			
			if(ii < row-2 && jj < column -2) // up right	
				constraints.push_back( new Constraint(getParticle(ii+2,jj), getParticle(ii,jj+2) ) );
		}	
	}	
}

/*--------------------------------------------------------------------
					Make Particles
--------------------------------------------------------------------*/
void make_particles(struct Particle *pVector, float4 *data_pointer, int row, int column, int width, int height)
{
    for(int index = 0; index < size; index++)
    {
	    int i = index%row;
	    int j = index/column;
	
	    float3 pos = make_float3(width * (i/(float)row), -height * (j/(float)column), 0);
	
	    if((j == 0 && i == 0) || (i == 0 && j == column-1))
        {
		    pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, getParticleInd(i,j, row), false);
        }
	    else
		    pVector[getParticleInd(i,j,row)] = Particle(pos, 1, data_pointer, getParticleInd(i,j, row), true);
    }
	
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
}

/*----------------------------------------------------------
					Delete VBO
----------------------------------------------------------*/
void deleteVBO(int clothNum)
{
    glBindBuffer( 1, vbo[clothNum]);
    glDeleteBuffers( 1, &(vbo[clothNum]));
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
        printf("texCoord = (%f, %f)\n", flagTexArray[ii].x, flagTexArray[ii].y);
    }

    printf("currIndex = %i\n", currIndex);
}


/*--------------------------------------------------------------------
					Initialize System
--------------------------------------------------------------------*/

void init_system(void)
{		
    vbo = (GLuint*)malloc(sizeof(GLuint) * numCloths);
    pVector = (Particle**)malloc(sizeof(Particle*) * numCloths);
    data_pointer = (float4**)malloc(sizeof(float4*) * numCloths);

	/* malloc cuda memory*/
    for(int ii = 0; ii < numCloths; ii++)
    {
        pVector[ii] = (struct Particle*)malloc(size * sizeof(struct Particle) );
        data_pointer[ii] = (float4*)malloc(sizeof(float4) * size);
		
	    /* initialize VBO */
        glGenBuffers(1, &(vbo[ii]));
        glBindBuffer(GL_ARRAY_BUFFER, vbo[ii]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * size, data_pointer, GL_DYNAMIC_DRAW);
	
	    make_particles(pVector[ii], data_pointer[ii], row, column, width, height); // create particles
		make_constraints();
	    /* unmap vbo */
        glUnmapBuffer(vbo[ii]);
    }
    /******************************
     * Flag texturing and meshing
     * ***************************/

    flagIndexArray = (unsigned int*)malloc(sizeof(unsigned int) * numTriangles * 3);
    flagTexArray = (float2*)malloc(sizeof(float2) * size);
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
}


/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

void draw_particles ( void )
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, flagTexId);

    for(int ii = 0; ii < numCloths; ii++)
    {
        glPushMatrix();
	        glColor3f(1.0, 1.0, 1.0);
            glTranslatef(ii*10, 0.0, 0.0);

            glBindBuffer(GL_ARRAY_BUFFER, vbo[ii]);
            glVertexPointer(4, GL_FLOAT, 0, 0);
            glBindBuffer(GL_ARRAY_BUFFER, texVbo);
            glTexCoordPointer(2, GL_FLOAT, 0, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            // Render flag mesh
            glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, 0);

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

            glDisableClientState(GL_VERTEX_ARRAY); 
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glPopMatrix(); 
    }

    glDisable(GL_TEXTURE_2D);
}

/*--------------------------------------------------------------------
					Draw Foreces
--------------------------------------------------------------------*/
void draw_forces ( void )
{}

/*----------------------------------------------------------------------
relates mouse movements to tinker toy construction
----------------------------------------------------------------------*/
void remap_GUI(struct Particle *pVector, float4 *data_pointer)
{	
    for(int index = 0; index < row*column; index++)
    {	
	    pVector[index].reset();
	    data_pointer[index] = make_float4(pVector[index].m_ConstructPos, 1);
    }
}

void step_func ( )
{
    for(int ii = 0; ii < numCloths; ii++)
    {
	    if ( dsim ){ // simulate
			verlet_simulation_step(pVector[ii], constraints, data_pointer[ii], vbo[ii], row, column, wind, numCloths);
	    }
	    else { // remap
		
	        // remove old 
		    deleteVBO(ii);

		    /* initialize VBO */
		    createVBO(ii);

		    /* map vbo in cuda */
            glBindBuffer(GL_ARRAY_BUFFER, vbo[ii]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * size, data_pointer[ii], GL_DYNAMIC_DRAW);
		
		    remap_GUI(pVector[ii], data_pointer[ii]);
		
		    /* unmap vbo */
            glUnmapBuffer(vbo[ii]);
	    }
    }
}
