// 
//  Particle.h
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#pragma once

#include "open_gl.hh"
#include "cuda_err.hh"
#include "cuda_helpers.hh"

#define DAMPING 0.01 // how much to damp the cloth simulation each frame

struct Particle
{
 __device__ __host__
	Particle(const float3 & ConstructPos, double mass, bool move) :
			m_ConstructPos(ConstructPos), m_Position(ConstructPos), m_Old(ConstructPos),
			m_normal(make_float3(0.0f)), m_forces(make_float3(0.0f)), m_mass(mass), m_movable(move) {}	
				
__device__ __host__ 
	void reset()
	{
		m_Position = m_Old = m_ConstructPos;
		m_forces = make_float3(0.0f);
	
	} // resets the position, velocity, and force
		
// ===============================
// = Add a force to the particle =
// ===============================

__device__ __host__	
	void addForce(float3 force){

		m_forces += force/m_mass;	
	}

__device__ __host__
	void updateVector(float3 new_pos){ /* updates the vector position */

		m_Position += new_pos;
	} // set vector

__device__ __host__	
	void updateNormal(float3 normal){ /* updates the vector normal */

		m_normal += normal;
	}

__device__ __host__	
	void step(float time){
		
		if(m_movable)
		{
			Vec3f temp = m_Position;
			m_Position = m_Position + (m_Position - m_Old)*(1.0-DAMPING) + m_forces * time;
			m_Old = temp;
			m_forces = Vec3f(0.0f); 				

		} // end if

	} // end step
	
	float3 m_ConstructPos; // original position
	float3 m_Position; // current position
	float3 m_Old; // previous integration position
	float3 m_normal; // "normal" of point
	float3 m_forces; // forces
	double m_mass; // mass of particle
	bool m_movable; // if movable or not

}; // end struct 