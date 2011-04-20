// 
//  data.hpp
//  P7_2
//  
//  Created by Timothy Luciani on 2011-04-17.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

#ifndef SERIALIZATION_DATA_HPP
#define SERIALIZATION_DATA_HPP

#include <string>

/// Structure to hold information about a single triangle.
struct m_triangle
{
	int test_int;
	float vertex_normals[3][4]; /* vertex normals */
//	int t[3]; /* texture coordinates */ 
	float vertices[3][4]; /* 3 coordinates describing the triangle */
};

struct m_float4 {
	
	/* custom float 4 */
	
	float x;
	float y;
	float z;
	float w;	
};

#endif // SERIALIZATION_DATA_HPP