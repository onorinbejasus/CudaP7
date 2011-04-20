// 
//  Sphere.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-12.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Shapes.h"
#include "GLUT/glut.h"

class Sphere : public Shape {
	
public:
	
	Sphere(float radius, Vec3f position, int shape)  
	{
		m_radius = radius; 
		m_position = position; 
		m_shape =shape; 
	}
	
	void draw(){
		
		glColor3f(0.0f, 1.0f, 0.0f); 
		/* draw a sphere around the point */

		glPushMatrix();
		glTranslatef(m_position[0], m_position[1], m_position[2]);
		glutWireSphere(m_radius, 30, 30);
		glPopMatrix();
	} // end draw
	
}; // end sphere class
