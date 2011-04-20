// 
//  Particle.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Particle.h"
#include <GLUT/glut.h>

#define DAMPING 0.01 // how much to damp the cloth simulation each frame

Particle::Particle(const Vec3f & ConstructPos, double mass, bool move) :
		m_ConstructPos(ConstructPos), m_Position(ConstructPos), m_Old(ConstructPos),
		m_normal(Vec3f(0.0f, 0.0f, 0.0f)), m_forces(Vec3f(0.0,0.0, 0.0)), m_mass(mass), m_movable(move) {}

Particle::~Particle(void) {}

void Particle::reset()
{
	m_Position = m_Old = m_ConstructPos;
	m_forces = Vec3f(0.0, 0.0, 0.0);
}

// ===============================
// = Add a force to the particle =
// ===============================
void Particle::addForce(Vec3f force){
	
	m_forces += force/m_mass;	
}

void Particle::updateVector(Vec3f new_pos){ /* updates the vector position */

	m_Position += new_pos;
}

void Particle::updateNormal(Vec3f normal){ /* updates the vector normal */

	m_normal += normal;
}

void Particle::step(float time)
{
	if(m_movable)
	{
		Vec3f temp = m_Position;
		m_Position = m_Position + (m_Position - m_Old)*(1.0-DAMPING) + m_forces * time;
		m_Old = temp;
		m_forces = Vec3f(0.0f); 				
	
	} // end if

} // end step

void Particle::draw()
{
	const double h = 0.005;
	glColor3f(1.f, 0.f, 0.f); 
	
	/* draw a sphere around the point */
	
	glPushMatrix();
	glTranslatef(m_Position[0], m_Position[1], m_Position[2]);
	glutWireSphere(h, 30, 30);
	glPopMatrix();

}
