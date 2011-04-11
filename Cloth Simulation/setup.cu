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

int row	  = 25;
int column = 25;

int width = 2;
int height = 2;

struct Particle* pVector;

struct Particle *draw;

int size = row * column;

extern void verlet_simulation_step(struct Particle* pVector, int row, int column);
extern int dsim;

struct Particle *getParticleCPU(int x, int y) { return &draw[y*row+x]; }
/* find the normal of a triangle */

float3 triangle_normal(float3 v1, float3 v2, float3 v3){
	
	float3 temp = cross(v2-v1, v3-v1);
	return normalize(temp);
}

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/
void free_data ( void )
{
	cudaFree(pVector);
}

void clear_data ( void )
{
	int ii;

	for(ii=0; ii<size; ii++){
		pVector[ii].reset();
	}
}

/*--------------------------------------------------------------------
					Make Particles
--------------------------------------------------------------------*/
void make_particles(struct Particle *pVector)
{
	for(int i = 0; i < row; i++){
		for(int j = 0; j < column; j++){
			
			float3 pos = make_float3(width * (i/(float)row), 0, -height * (j/(float)column));
			if(j == 0)
				pVector[j * row + i] = Particle (pos, 1, false);
			else
				pVector[j * row + i] = Particle (pos, 1, true);
		}
	}
	
} // end make particles

/*--------------------------------------------------------------------
					Update Normals
--------------------------------------------------------------------*/

void calculate_normals(){
	
	for(int ii = 0; ii < row-1; ii++){
		
		for(int jj = 0; jj < column-1; jj++){
			
			 // Particle 1 = ii+1, jj
			// Particle 2 = ii, jj
			// Particle 3 = ii, jj+1
			
			/* calculate normal of triangle */
			
			float3 normal = triangle_normal(getParticleCPU(ii+1,jj)->m_Position, getParticleCPU(ii,jj)->m_Position, getParticleCPU(ii,jj+1)->m_Position);
			
			/* add component to triangle */
			
			getParticleCPU(ii+1,jj)->updateNormal(normal); 
			getParticleCPU(ii,jj)->updateNormal(normal);
		 	getParticleCPU(ii+1,jj+1)->updateNormal(normal);
		
			/* calculate normal of triangle */
		
			normal = triangle_normal(getParticleCPU(ii+1,jj+1)->m_Position, getParticleCPU(ii+1,jj)->m_Position, getParticleCPU(ii,jj+1)->m_Position);
		
			/* add component to triangle */
		
			getParticleCPU(ii+1,jj+1)->updateNormal(normal);
			getParticleCPU(ii+1,jj)->updateNormal(normal);
	 		getParticleCPU(ii,jj+1)->updateNormal(normal);
		
		} // end for jj
	} // end for ii
	
}

/*--------------------------------------------------------------------
					Initialize System
--------------------------------------------------------------------*/

void init_system(void)
{	
	draw = (struct Particle*)malloc(size * sizeof(struct Particle) );
	
	/* temporary particle */
	struct Particle *temp_p;
	temp_p = (struct Particle*)malloc(size * sizeof(struct Particle) );
	
	/* malloc cuda memory*/
	cudaMalloc( (void**)&pVector, size * sizeof(struct Particle) );
	
	/* create and copy */
	
	make_particles(temp_p); // create particles
	cudaMemcpy(pVector, temp_p, size * sizeof(struct Particle), cudaMemcpyHostToDevice);
	
	cudaMemcpy(draw, pVector, size * sizeof(struct Particle), cudaMemcpyDeviceToHost);
	
	free(temp_p);
}

/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

void draw_particles ( void )
{
	for(int ii=0; ii< size; ii++)
	{
		draw[ii].draw();
	}
}

/*--------------------------------------------------------------------
					Draw Foreces
--------------------------------------------------------------------*/
void draw_forces ( void )
{}

/*----------------------------------------------------------------------
relates mouse movements to tinker toy construction
----------------------------------------------------------------------*/

void remap_GUI()
{	
	int ii;
	
	for(ii=0; ii<size; ii++)
	{
		draw[ii].reset();
	}
	
	cudaMemcpy(pVector, draw, size * sizeof(struct Particle), cudaMemcpyHostToDevice);
}

void step_func ( )
{
	if ( dsim ){
		verlet_simulation_step(pVector, row, column);
		
		cudaMemcpy(draw, pVector, size * sizeof(struct Particle), cudaMemcpyDeviceToHost);
	}
	else {remap_GUI();}
}