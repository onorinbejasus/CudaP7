// 
//  Constraint.h
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#pragma once

#include "Particle.h"
#include "cmath"
#include "open_gl.hh"
#include "cuda_err.hh"
#include "cuda_helpers.hh"

struct Constraint{
		
	struct Particle *p1, *p2; // two particle linked
	float m_rest; // rest position

__device__ __host__	
	Constraint(){} // default constructor

__device__ __host__		
	Constraint(struct Particle *part1, struct Particle *part2) : p1(part1), p2(part2){

		float3 diff = part2->m_Position - part1->m_Position;
		m_rest = sqrt(diff * diff);
	}
	
__device__ __host__	
	void satisfy(){

		/* check distance between particles against rest distance, then correct */ 

		Vec3f p1_to_p2 = p2->m_Position - p1->m_Position; 
		float current_distance = sqrt(p1_to_p2 * p1_to_p2);
		Vec3f correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
		Vec3f correctionVectorHalf = correctionVector*0.5; 

		if(current_distance > m_rest){

			p1->updateVector(correctionVectorHalf); 
			p2->updateVector(-correctionVectorHalf);
		} // end if

		if(!p1->m_movable)
			p1->m_Position = p1->m_ConstructPos;

		if(!p2->m_movable)
			p2->m_Position = p2->m_ConstructPos;

	} // end satisfy // satisfy constaints
	
}; // end constraint class