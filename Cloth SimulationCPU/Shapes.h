// 
//  Shapes.h
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-12.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#pragma once
#include "gfx/vec3.h"

#define SPHERE 0

class Shape {

public:
	float m_radius;
	Vec3f m_position;
	int m_shape;
		
	Shape(){}; // default constructor
	
	virtual void draw() = 0;
	
}; // end shape