// 
//  Constraint.h
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#pragma once

#include "Constraint.h"
#include "Particle.h"
#include "Particle.h"
#include "gfx/vec3.h"

#include "cmath"

class Constraint{
	
private:
	
	Particle *p1, *p2; // two particle linked
	float m_rest; // rest position
	
public:
	Constraint(){} // default constructor
	Constraint(Particle *part1, Particle *part2); // constructor
	~Constraint(); // destructor
	void satisfy(); // satisfy constaints
	
}; // end constraint class