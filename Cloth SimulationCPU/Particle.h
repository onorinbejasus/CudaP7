// 
//  Particle.h
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#pragma once
#include "gfx/vec3.h"

class Particle
{
public:

	Particle(const Vec3f & ConstructPos, double mass, bool move);
	virtual ~Particle(void);

	void reset(); // resets the position, velocity, and force
	void draw(); // draws the circle
	void addForce(Vec3f force); // add a force to the particle
	void updateVector(Vec3f vector); // set vector
	void updateNormal(Vec3f normal); // set vector
	void step(float time);
	
	Vec3f m_ConstructPos; // original position
	Vec3f m_Position; // current position
	Vec3f m_Old; // previous integration position
	Vec3f m_normal; // "normal" of point
	Vec3f m_forces; // forces
	double m_mass; // mass of particle
	bool m_movable; // if movable or not
};
