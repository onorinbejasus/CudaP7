// 
//  Constraint.cpp
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-03-11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#include "Constraint.hh"

Constraint::Constraint(Particle *part1, Particle *part2) : p1(part1), p2(part2){
	
	float3 diff = part2->m_Position - part1->m_Position;
	m_rest = length(diff);
}

void Constraint::satisfy(float4 *data_pointer){

	/* check distance between particles against rest distance, then correct */ 

	float3 p1_to_p2 = p2->m_Position - p1->m_Position; 
	float current_distance = length(p1_to_p2);
	float3 correctionVector = p1_to_p2*(1 - m_rest/current_distance); 
	float3 correctionVectorHalf = correctionVector*0.5; 
		
	if(current_distance > m_rest){
		
		p1->updateVector(correctionVectorHalf, data_pointer); 
		p2->updateVector(-correctionVectorHalf, data_pointer);
	} // end if
	
	// pinned particle
	
	if(!p1->m_movable){
		p1->m_Position = p1->m_ConstructPos;
		data_pointer[p1->m_index] = make_float4(p1->m_ConstructPos,1);
	}
	if(!p2->m_movable){
		p2->m_Position = p2->m_ConstructPos;
		data_pointer[p2->m_index] = make_float4(p2->m_ConstructPos,1);
	}
} // end satisfy