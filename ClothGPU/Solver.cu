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

#include <cutil.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <cstdlib>
#include <cstdio>

#define TIME_STEP 0.5*0.5 // how large time step each particle takes each frame
#define CONSTRAINT_ITERATIONS 25 // how many iterations of constraint satisfaction each frame (more is rigid, less is soft)

extern void createVBO(int numCloth);
extern void deleteVBO(int numCloth);

const float3 gravity = make_float3(0.0f, -0.15f, 0.0f);
static const int threadsPerBlock = 64;

__device__ __host__
int getParticle(int x, int y, int row){ return y*row+x; }

/* find the normal of a triangle */

__device__
float3 triangle_normal(float3 v1, float3 v2, float3 v3){ return ( cross(v2-v1, v3-v1) ); }

__device__
float3 triangle_normal(float4 v1, float4 v2, float4 v3)
{ 
    float3 myV1 = make_float3(v1.x, v1.y, v1.z);
    float3 myV2 = make_float3(v2.x, v2.y, v2.z);
    float3 myV3 = make_float3(v3.x, v3.y, v3.z);

    return (cross(myV3-myV1, myV2-myV1)); 
}


/* apply the wind force to the cloth */

__device__
float3 windForce(struct Particle *pVector, float3 windDir, int x, int y, int row)
{
	float3 normal = triangle_normal(pVector[getParticle(x+1,y,row)].m_Position, 
		pVector[getParticle(x,y,row)].m_Position, pVector[getParticle(x,y+1,row)].m_Position);
	
	float3 d = normalize(normal);
	float3 force = normal * dot(d,windDir);
	
	normal = triangle_normal(pVector[getParticle(x+1,y+1,row)].m_Position, 
	pVector[getParticle(x+1,y,row)].m_Position, pVector[getParticle(x,y+1,row)].m_Position);

	d = normalize(normal);
	force += normal * dot(d,windDir);

	return force;
}

__global__
void add_force(struct Particle *pVector, float3 gravity, bool wind, int row, int column){	
	
	// calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	int x = index%row;
	int y = index/column;

	pVector[index].step(TIME_STEP);
	
	/* gravity */
	pVector[index].addForce(gravity * TIME_STEP);
	
	if(wind && y < (column -1) && x < (row-1)){
		float3 windDir = make_float3(0.3f, 0.3f, 0.2f);
	
		/* wind */
		pVector[index].addForce( windForce(pVector, windDir, x, y, row) * 20 );
	
	}
}

__global__
void satisfy(struct Particle *pVector, float4 *data_pointer, int row, int column){
	
	//calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	 
	int ii = index%row;
	int jj = index/column;

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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);
			}

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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);
				
			}

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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);
				
			}		
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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);
				
			}	
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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);
				
			}		
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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);
				
			}
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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);
				
			}		
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

				pVector[p1].updateVector(correctionVectorHalf, data_pointer); 
				pVector[p2].updateVector(-correctionVectorHalf, data_pointer);

			} // end if

			if(!pVector[p1].m_movable){
				
				pVector[p1].m_Position = pVector[p1].m_ConstructPos;
				data_pointer[p1] = make_float4(pVector[p1].m_ConstructPos, 1);
			}
			if(!pVector[p2].m_movable){
				
				pVector[p2].m_Position = pVector[p2].m_ConstructPos;
				data_pointer[p2] = make_float4(pVector[p2].m_ConstructPos, 1);	
			}	
		}	
	}
}

__global__
void calculate_flag_normals(float4 *data_pointer, float3 *flagNorms, int row, int column)
{
	// calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	int x = index%row;
	int y = index/column;

    float3 currNorm = make_float3(0.0f, 0.0f, 0.0f);

    if(x == 0 && y == 0)
    {
        // Top Left
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column], data_pointer[index+1]);
    }
    else if(x == (column-1) && y == 0)
    {
        // Top Right
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-1], data_pointer[index+column-1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column-1], data_pointer[index+column]);
    }
    else if(x == 0 && y == (column-1))
    {
        // Bottom Left
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column+1], data_pointer[index-column]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+1], data_pointer[index-column+1]);
    }
    else if(x == (row-1) && y == (column-1))
    {
        // Bottom Right
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column], data_pointer[index-1]);
    }
    else if(y == 0)
    {
        // Top row
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-1], data_pointer[index+column-1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column-1], data_pointer[index+column]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column], data_pointer[index+1]);
    }
    else if(y == (column-1))
    {
        // Bottom row
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column], data_pointer[index-1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column+1], data_pointer[index-column]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+1], data_pointer[index-column+1]);
    }
    else if(x == 0)
    {
        // Left column
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column+1], data_pointer[index-column]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+1], data_pointer[index-column+1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column], data_pointer[index+1]);
    }
    else if(x == (row-1))
    {
        // Right column 
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column], data_pointer[index-1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-1], data_pointer[index+column-1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column-1], data_pointer[index+column]);
    }
    else
    {
        // Middle vertex that touches six faces
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-1], data_pointer[index+column-1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column-1], data_pointer[index+column]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+column], data_pointer[index+1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column], data_pointer[index-1]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index-column+1], data_pointer[index-column]);
        currNorm += triangle_normal(data_pointer[index], data_pointer[index+1], data_pointer[index-column+1]);
    }

    flagNorms[index] = normalize(currNorm);
}

void verlet_simulation_step(struct Particle* pVector, float4 *data_pointer, GLuint vbo, float3 *flagNorms, GLuint nVbo, bool wind, int row, int column, int numCloth){
				
	/* set up number of threads to run */	
	int totalThreads = row * column;
	int nBlocks = totalThreads/threadsPerBlock;
	nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;
	
	/* apply wind and gravity forces */	
	add_force<<<nBlocks, threadsPerBlock>>>(pVector, gravity, wind, row, column);
		
	/* map vbo in cuda */
	cudaGLMapBufferObject((void**)&data_pointer, vbo);
				
 	satisfy<<<nBlocks, threadsPerBlock>>>(pVector, data_pointer, row, column);

	/* unmap vbo */
	cudaGLUnmapBufferObject(vbo);

    cudaGLMapBufferObject((void**)&flagNorms, nVbo);

    calculate_flag_normals<<<nBlocks, threadsPerBlock>>>(data_pointer, flagNorms, row, column);

    cudaGLUnmapBufferObject(nVbo);
				
} // end sim step
