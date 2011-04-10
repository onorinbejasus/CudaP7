/************************************
Author: Timothy Luciani
File: Mesh.cpp
************************************/
#include "Mesh.h"

using namespace std;

Mesh::Mesh(){}; /* default constructor */

Mesh::~Mesh(){} /* deconstructor */

bool Mesh::loadModelData(const char *filename){

	string comment; /* comment string */
	struct Vertex v; /* vertex struct */
	struct Normal n; /* normals struct */
	struct Triangle t; /* triangle struct */
	struct Texture te; /* texture struct */
	
	/* file stream to read in the texture */
	ifstream input (filename, ios::in); 
	
	/* if file cannot be opened */
	if(input.fail())
		return false;

	number_verts = 0; /* number of vertices */

	/* start the vectors at 1 so that the it matches the obj format */
	
	vertices.push_back(v); /* index 0 */ 
	normals.push_back(n); /* index 0 */
	textures.push_back(te); /* index 0 */
	
	/* increment number of vertices/normals */

	number_verts++;
	number_normals++;
	number_textures++;
	
	char *peek = new char[2]; /* peek string */

	/* iterate over file */

	while(input.readsome(peek, 2) != input.eof()) /* look at the next 2 characters */
	{	 		
		if( strncmp(peek,"#", 1) == 0) { /* comment */
			
			getline(input, comment);
			cout << comment << endl;
		
		}else if (strncmp(peek, "  ", 2) == 0){
		
			cout << "blank line" << endl;
			
		}else if( strncmp(peek,"v ", 2) == 0){ /* vertex */
			
			number_verts++;  /* increment number of verts */
			string verts; /* line */
			char temp[128]; /* tempory c string */
			
			getline(input, verts); /* get line of vertices */
			strcpy(temp, verts.c_str()); /* copy to c string */
			
			/* get (X,Y,Z) */
			sscanf(temp, "%f %f %f", &v.location[0], &v.location[1], &v.location[2]);
			v.location[3] = 1; /* W */

			vertices.push_back(v); /* add to vector */
		
		}else if( strncmp(peek,"vn", 2) == 0){ /* vertex normal */

			number_normals++;  /* increment number of normals */
			string norms; /* line */
			char temp[128]; /* tempory c string */
			
			getline(input, norms); /* get line of vertices */
			strcpy(temp, norms.c_str()); /* copy to c string */
			
			/* get (X,Y,Z) */
			sscanf(temp, "%f %f %f", &n.normal[0], &n.normal[1], &n.normal[2]);
			n.normal[3] = 1; /* W */
		
			normals.push_back(n); /* add to vector */
		
		}else if( strncmp(peek,"vt", 2) == 0){  /* vertex texture */
		
			number_textures++;  /* increment number of normals */
			string texts; /* line */
			char temp[128]; /* tempory c string */
			
			getline(input, texts); /* get line of vertices */
			strcpy(temp, texts.c_str()); /* copy to c string */
			
			/* get (X,Y,Z) */
			sscanf(temp, "%f %f %f", &te.tex_coord[0], &te.tex_coord[1], &te.tex_coord[2]);
		
			textures.push_back(te); /* add to vector */
		
		}
		else if( strncmp(peek, "f ", 2) == 0){ /* face */
		
			number_triangles++; /* increment number of triangles */
		
			string temp; /* temporary string */
			char values[128]; /* hold the string read in */
			getline(input, temp); /* get the line */
			strcpy(values, temp.c_str()); /* convert to c string */	
			
			int vert[3]; int norm[3]; int tex[3]; /* hold the vertice, tex and normals */
			
			/* get (X,Y,Z) */
			
			if(number_textures > 1){ /* if textures were given */
				sscanf(values, "%d/%d/%d %d/%d/%d %d/%d/%d", &vert[0], &tex[0], &norm[0], 
					&vert[1], &tex[1], &norm[1], &vert[2], &tex[2], &norm[2]);
			
			
//			cout << vert[0] << "  " << tex[0] << " " << norm[0] << endl;
//			cout << vert[1] << "  " << tex[1] << " " << norm[1] << endl;
//			cout << vert[2] << "  " << tex[2] << " " << norm[2] << endl;
			
			}else{	
				sscanf(values, "%d/%d %d/%d %d/%d", &vert[0], &norm[0], 
					&vert[1], &norm[1], &vert[2], &norm[2]);
			}
			
			/* assign vertices to triangle */
		
			t.vertices[0][0] = vertices[vert[0]].location[0];
			t.vertices[0][1] = vertices[vert[0]].location[1];
			t.vertices[0][2] = vertices[vert[0]].location[2];
			t.vertices[0][3] = 1.0;
			
			t.vertices[1][0] = vertices[vert[1]].location[0];
			t.vertices[1][1] = vertices[vert[1]].location[1];
			t.vertices[1][2] = vertices[vert[1]].location[2];
			t.vertices[1][3] = 1.0;

			t.vertices[2][0] = vertices[vert[2]].location[0];
			t.vertices[2][1] = vertices[vert[2]].location[1];
			t.vertices[2][2] = vertices[vert[2]].location[2];
			t.vertices[2][3] = 1.0;
			
			/* assign normals to triangle */
			
			t.vertex_normals[0][0] = normals[norm[0]].normal[0];
			t.vertex_normals[0][1] = normals[norm[0]].normal[1];
			t.vertex_normals[0][2] = normals[norm[0]].normal[2];
			t.vertex_normals[0][3] = 1.0;
			
			t.vertex_normals[1][0] = normals[norm[1]].normal[0];
			t.vertex_normals[1][1] = normals[norm[1]].normal[1];
			t.vertex_normals[1][2] = normals[norm[1]].normal[2];
			t.vertex_normals[1][3] = 1.0;

			t.vertex_normals[2][0] = normals[norm[2]].normal[0];
			t.vertex_normals[2][1] = normals[norm[2]].normal[1];
			t.vertex_normals[2][2] = normals[norm[2]].normal[2];
			t.vertex_normals[2][3] = 1.0;
			
			if(number_textures > 1){ /* assign textures to triangle */
		/*	
				t.t[0][0] = textures[tex[0]].tex_coord[0];
				t.t[0][1] = textures[tex[0]].tex_coord[1];
				t.t[0][2] = textures[tex[0]].tex_coord[2];
			
				t.t[1][0] = textures[tex[1]].tex_coord[0];
				t.t[1][1] = textures[tex[1]].tex_coord[1];
				t.t[1][2] = textures[tex[1]].tex_coord[2];

				t.t[2][0] = textures[tex[2]].tex_coord[0];
				t.t[2][1] = textures[tex[2]].tex_coord[1];
				t.t[2][2] = textures[tex[2]].tex_coord[2];			
		*/	}
			
			triangles.push_back(t); /* add triangles to vector */
								
		}
		
	} /* end for loop */
	
	cout << "done!" << endl;
	
//	cout << number_verts << endl;
	
	return true;
	
} /* end load model data */
		
		