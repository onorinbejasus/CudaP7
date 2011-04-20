// 
//  Solver.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-12.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Particle.hh"
#include "Constraint.hh"

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <vector>

#define TIME_STEP 0.5*0.5 // how large time step each particle takes each frame
#define CONSTRAINT_ITERATIONS 25 // how many iterations of constraint satisfaction each frame (more is rigid, less is soft)

extern void createVBO(int numCloth);
extern void deleteVBO(int numCloth);

const float3 gravity = make_float3(0.0f, -0.15f, 0.0f);

float3 triangle_normal(float3 v1, float3 v2, float3 v3){ return ( cross(v2-v1, v3-v1) ); }
int getParticle(int x, int y, int row){ return y*row+x; }

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

void verlet_simulation_step(struct Particle* pVector, std::vector<Constraint*> constraints, 
										float4 *data_pointer, GLuint vbo, int row, int column, bool wind, int numCloth){
	
	for(int ii = 0; ii < row * column; ii++){
		pVector[ii].step(TIME_STEP);
		
		pVector[ii].addForce(gravity * TIME_STEP);
		
		int x = ii%row;
		int y = ii/column;
		
		if(wind && y < (column -1) && x < (row-1)){
			
			float3 windDir = make_float3(0.3f, 0.3f, 0.2f);	
			/* wind */
			pVector[ii].addForce( windForce(pVector, windDir, x, y, row) * 10 );

		}
		
	} // end for
	
	// remove old 
	deleteVBO(numCloth);
	
	/* initialize VBO */
	createVBO(numCloth);
	
	/* map vbo */
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * row * column, data_pointer, GL_DYNAMIC_DRAW);
	
	for(int ii = 0; ii < CONSTRAINT_ITERATIONS; ii++){
		
		for(int jj = 0; jj < constraints.size(); jj++){
			
			constraints[jj]->satisfy(data_pointer);
			
		} // end for jj	
	}	// end for ii
	
} // end sim step
