/************************************
Author: Timothy Luciani
File: Mesh.h
************************************/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include "data.hh"
#include <GLUT/glut.h>

typedef struct float4 {
	
	float x;
	float y;
	float z;
	float w;
	
};
typedef struct float3 {
	
	float x;
	float y;
	float z;	
};


class Mesh {
	
	protected:
		
		/* structure to hold vertices */
		struct Vertex {

			float4 location; /* (X,Y,Z,W) */
		};
		
		/* structure to hold normals */
		
		struct Normal {
		
		 	float4 normal; /* (X,Y,Z,W) */
		}; 
				
		/* structure that stores the material */
		struct Material {

			GLfloat ambient[4], diffuse[4], specularity[4], emissive[4]; /* lighting coefficients */
			GLfloat shininess; /* shininess */
			GLuint texture; 
			char *tex_filename; /* file name of the texture */
		};
		
		/* structure that stores textures */
		
		struct Texture {
		
			float3 tex_coord; /* stores the tex coordinate for the triangle */
		};
		
		public:
	
		bool loadModelData(const char filename[18], std::vector<struct m_triangle> *mesh);
		Mesh(); /* default constructor */
		~Mesh(); /* deconstructor */
		
		int number_verts; /* number of total vertices */
		std::vector<Vertex> vertices; /* verctor of vertices */
		
		int number_triangles; /* number of triangles */
	
		int number_normals; /* number of normals */
		std::vector<Normal> normals; /* vector of normals */
		
		int number_textures; /* number of textures */
		std::vector<Texture> textures; /* vector of textures */
		
		
}; /* end texture class */