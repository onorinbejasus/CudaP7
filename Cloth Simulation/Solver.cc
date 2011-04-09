// 
//  Solver.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-12.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Particle.h"
#include "Constraint.h"
#include "Shapes.h"

#include <vector>
#include <cstdlib>
#include <cstdio>

#define TIME_STEP 0.5*0.5 // how large time step each particle takes each frame
#define CONSTRAINT_ITERATIONS 100 // how many iterations of constraint satisfaction each frame (more is rigid, less is soft)

const Vec3f gravity(0.0f, -0.15f, 0.0f);

void verlet_simulation_step(std::vector<Particle*> pVector, std::vector<Constraint*> constraints, Shape *shape){
	
	for(int ii = 0; ii < pVector.size(); ii++){
			pVector[ii]->addForce(gravity * TIME_STEP);
	} // end for
	
	for(int ii = 0; ii < CONSTRAINT_ITERATIONS; ii++){
		
		for(int jj = 0; jj < constraints.size(); jj++){
			
			constraints[jj]->satisfy();
			
		} // end for jj	
	}	// end for ii
		
	for(int ii = 0; ii < pVector.size(); ii++){
	
		Vec3f vec = (pVector[ii]->m_Position) - shape->m_position; // position in respect to center
		float len = sqrt(vec * vec); // length
		
		if(len < shape->m_radius) // if inside sphere
			pVector[ii]->updateVector(vec/len * (shape->m_radius - len));	
		
		pVector[ii]->step(TIME_STEP);
	}
		
	
} // end sim step
