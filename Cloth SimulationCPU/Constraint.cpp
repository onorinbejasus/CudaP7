// 
//  Constraint.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Constraint.h"

Constraint::Constraint(Particle *part1, Particle *part2) : p1(part1), p2(part2){
	
	Vec3f diff = part2->m_Position - part1->m_Position;
	m_rest = sqrt(diff * diff);
}

void Constraint::satisfy(){

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

} // end satisfy