// 
//  Particle.hh
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#pragma once

#include "open_gl.hh"
#include "cuda_helpers.hh"

#define DAMPING 0.1  // how much to damp the cloth simulation each frame

struct Particle
{
	Particle(){}
	
	Particle(const float3 & ConstructPos, float mass, float4 *data_pointer, int index, bool move) :
			m_ConstructPos(ConstructPos), m_Position(ConstructPos), m_Old(ConstructPos),
			m_normal(make_float3(0.0f)), m_forces(make_float3(0.0f)), m_mass(mass), m_index(index), m_movable(move) {
				
				data_pointer[index] = make_float4(m_ConstructPos, 1);
			}	

	void draw()
	{
		const double h = 0.005;
		glColor3f(1.f, 0.f, 0.f); 

		/* draw a sphere around the point */

		glPushMatrix();
		glTranslatef(m_Position.x, m_Position.y, m_Position.z);
		glutWireSphere(h, 30, 30);
		glPopMatrix();

	}
					
	void reset()
	{
		m_Position = m_Old = m_ConstructPos;
		m_normal = make_float3(0.0f);
		m_forces = make_float3(0.0f);
	
	} // resets the position, velocity, and force
		
// ===============================
// = Add a force to the particle =
// ===============================

	void addForce(float3 force){

		m_forces += force/m_mass;	
	}

	void updateVector(float3 new_pos, float4 *data_pointer){ /* updates the vector position */

		m_Position += new_pos;
		data_pointer[m_index] = make_float4(m_Position, 1);
		
	} // set vector

	void updateNormal(float3 normal){ /* updates the vector normal */

		m_normal += normal;
	}

	void step(float time){
		
		if(m_movable)
		{
			float3 temp = m_Position;
			m_Position = m_Position + (m_Position - m_Old)*(1.0-DAMPING) + m_forces * time;
			m_Old = temp;
			m_forces = make_float3(0.0f); 				

		} // end if

	} // end step
	
	uint m_index;
	float3 m_ConstructPos; // original position
	float3 m_Position; // current position
	float3 m_Old; // previous integration position
	float3 m_normal; // "normal" of point
	float3 m_forces; // forces
	float m_mass; // mass of particle
	bool m_movable; // if movable or not

}; // end struct 
