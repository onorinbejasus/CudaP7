// 
//  Solver.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-12.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Particle.hh"
#include "Constraint.hh"

#include "open_gl.hh"
#include "cuda_err.hh"
#include "cuda_helpers.hh"

#include <vector>
#include <cstdlib>
#include <cstdio>

#define TIME_STEP 0.5*0.5 // how large time step each particle takes each frame
#define CONSTRAINT_ITERATIONS 15 // how many iterations of constraint satisfaction each frame (more is rigid, less is soft)

const float3 gravity = make_float3(0.0f, -0.15f, 0.0f);

//__global__
void verlet_simulation_step(struct Particle* pVector, struct Constraint* constraints, int size, int num_con){
	
	for(int ii = 0; ii < size; ii++){
			pVector[ii].addForce(gravity * TIME_STEP);
	} // end for
	
	for(int ii = 0; ii < CONSTRAINT_ITERATIONS; ii++){
		
		for(int jj = 0; jj < num_con; jj++){
			
			constraints[jj].satisfy();
			
		} // end for jj	
	}	// end for ii
		
	for(int ii = 0; ii < size; ii++){
		
		pVector[ii].step(TIME_STEP);
	}
		
} // end sim step
