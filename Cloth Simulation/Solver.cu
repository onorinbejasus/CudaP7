// 
//  Solver.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-12.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Particle.hh"

#include "open_gl.hh"
#include "cuda_err.hh"
#include "cuda_helpers.hh"
#include <cuda.h>

#include <vector>
#include <cstdlib>
#include <cstdio>

#define TIME_STEP 0.5*0.5 // how large time step each particle takes each frame
#define CONSTRAINT_ITERATIONS 15 // how many iterations of constraint satisfaction each frame (more is rigid, less is soft)

const float3 gravity = make_float3(0.0f, -0.15f, 0.0f);

__device__ __host__
int getParticle(int x, int y, int row){ return y*row+x; }

static const int threadsPerBlock = 512;

__global__
void add_force(struct Particle *pVector, float3 gravity, int row, int column){	
	
	// calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	pVector[index].step(TIME_STEP);
	pVector[index].addForce(gravity * TIME_STEP);
}

__global__
void satisfy(struct Particle *pVector, int row, int column){
	
	// //calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// 
	float ii = index%row;
	float jj = index/column;

	for(int i = 0; i < CONSTRAINT_ITERATIONS; i++){		
		
		if(ii < row-1){ // to the right

			int p1 = getParticle(ii,jj, row);
			int p2 = getParticle(ii+1, jj, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos;
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;

		}
		if(jj < column -1){ // below	

			int p1 = getParticle(ii,jj, row);
			int p2 = getParticle(ii, jj+1, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos;
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;

		}
		if(ii < row-1 && jj < column -1){ // down right

			int p1 = getParticle(ii,jj, row);
			int p2 = getParticle(ii+1, jj+1, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos;
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;		
		}
		if(ii < row-1 && jj < column -1){ // up right	

			int p1 = getParticle(ii+1,jj, row);
			int p2 = getParticle(ii, jj+1, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos;
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;	
		}
		/* neighbor's neighbors */

	 	if(ii < row-2){ // to the right

			int p1 = getParticle(ii,jj, row);
			int p2 = getParticle(ii+2, jj, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos; 
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;		
		}
		if(jj < column -2){ // below	

			int p1 = getParticle(ii,jj, row);
			int p2 = getParticle(ii, jj+2, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos;
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
		}

		if(ii < row-2 && jj < column -2){ // down right

			int p1 = getParticle(ii,jj, row);
			int p2 = getParticle(ii+2, jj+2, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos;
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
		}

		if(ii < row-2 && jj < column -2){ // up right	

			int p1 = getParticle(ii+2,jj, row);
			int p2 = getParticle(ii, jj+2, row);

			float3 diff = pVector[p1].m_ConstructPos - pVector[p2].m_ConstructPos;
			float m_rest = length(diff);

			float3 p1_to_p2 = pVector[p2].m_Position - pVector[p1].m_Position; 
			float current_distance = length(p1_to_p2);
			float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
			float3 correctionVectorHalf = correctionVector*0.5; 

			if(current_distance > m_rest){

				pVector[p1].updateVector(correctionVectorHalf); 
				pVector[p2].updateVector(-correctionVectorHalf);

			} // end if

			if(!pVector[p1].m_movable)
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;

			if(!pVector[p2].m_movable)
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;	
		}	
	}
}

void verlet_simulation_step(struct Particle* pVector, int row, int column){
	
	struct Particle *temp;
	temp = (struct Particle*)malloc(row * column * sizeof(struct Particle));
				
	/* set up number of threads to run */	
	int totalThreads = row * column;
	int nBlocks = totalThreads/threadsPerBlock;
	nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
		
	add_force<<<nBlocks, threadsPerBlock>>>(pVector, gravity, row, column);
	
	cudaMemcpy(temp, pVector, row * column * sizeof(struct Particle), cudaMemcpyDeviceToHost);
	
	cudaThreadSynchronize();

// 	satisfy(temp, row, column);
	
//	cudaMemcpy(pVector, temp, row * column * sizeof(struct Particle), cudaMemcpyHostToDevice);
		
	free(temp);	
		
 	satisfy<<<nBlocks, threadsPerBlock>>>(pVector, row, column);
				
} // end sim step
