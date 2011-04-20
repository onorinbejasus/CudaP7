// 
//  Constraint.h
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#pragma once

#include "Particle.hh"

#include "cmath"

class Constraint{
	
private:
	
	Particle *p1, *p2; // two particle linked
	float m_rest; // rest position
	
public:
	Constraint(){} // default constructor
	Constraint(struct Particle *part1, struct Particle *part2); // constructor
	~Constraint(); // destructor
	void satisfy(float4 *data_pointer); // satisfy constaints
	
}; // end constraint class