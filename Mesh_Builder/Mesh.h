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

#include <GLUT/glut.h>

class Mesh {
	
	protected:
		
		/* structure to hold vertices */
		struct Vertex {

			GLfloat location[4]; /* (X,Y,Z,W) */
		};
		
		/* structure to hold normals */
		
		struct Normal {
		
			GLfloat normal[4]; /* (X,Y,Z,W) */
		}; 
		
		/* structure that holds triangles */
		struct Triangle {

			GLfloat vertex_normals[3][4]; /* vertex normals */
			int t[3]; /* texture coordinates */ 
			GLfloat vertices[3][4]; /* 3 coordinates describing the triangle */
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
		
			GLfloat tex_coord[3]; /* stores the tex coordinate for the triangle */
		};
		
		public:
	
		bool loadModelData(const char *filename);
		Mesh(); /* default constructor */
		~Mesh(); /* deconstructor */
		
		int number_verts; /* number of total vertices */
		std::vector<Vertex> vertices; /* verctor of vertices */
		
		int number_triangles; /* number of triangles */
		std::vector<Triangle> triangles; /* vector of triangles */
	
		int number_normals; /* number of normals */
		std::vector<Normal> normals; /* vector of normals */
		
		int number_textures; /* number of textures */
		std::vector<Texture> textures; /* vector of textures */
		
		
}; /* end texture class */