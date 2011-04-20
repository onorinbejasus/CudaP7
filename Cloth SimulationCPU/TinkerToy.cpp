// TinkerToy.cpp : Defines the entry point for the console application.

#include "Particle.h"
#include "Constraint.h"
#include "imageio.h"
#include "Sphere.h"

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <GLUT/glut.h>

#include "gfx/geom3d.h"

#include <math.h>

/* colors */

#define BLACK   0
#define RED     1
#define YELLOW  2
#define MAGENTA 3
#define GREEN   4
#define CYAN    5
#define BLUE    6
#define GREY    7
#define WHITE   8

#define MAXSAMPLES 100

/* external definitions (from solver) */

extern void verlet_simulation_step(std::vector<Particle*> pVector, std::vector<Constraint*> constraints, Shape *shape);
	
/* global variables */

static int N;
static float d;
static int dsim;

// static Particle *pList;
static std::vector<Particle*> pVector;
static std::vector<Constraint*> constraints;
static std::vector<Shape*> shapes;

static int win_id;
static int win_x, win_y;

static int row	  = 19;
static int column = 19;

static int size = row * column;

static int width = 1;
static int height = 1;

static int x_camera = 0, y_camera = 0, z_camera = 10;
static int lookAtX = 0, lookAtY = 0, lookAtZ = -1;

static bool visible = false;

/* spring constants */

Particle *getParticle(int x, int y){ return pVector[y*row+x]; }

/*----------------------------------------------------------------------
free/clear/allocate simulation data
----------------------------------------------------------------------*/

static void free_data ( void )
{
	pVector.clear(); // empty particles
	constraints.clear(); // empty constraints
}

static void clear_data ( void )
{
	int ii, size = pVector.size();

	for(ii=0; ii<size; ii++){
		pVector[ii]->reset();
	}
}

/*--------------------------------------------------------------------
					Make Particles
--------------------------------------------------------------------*/
static void make_particles(void)
{
	for(int i = 0; i < row; i++){
		for(int j = 0; j < column; j++){
			
			Vec3 pos = Vec3(width * (i/(float)row), 0, -height * (j/(float)column));
			if(j == 0)
				pVector[j * row + i] = new Particle (pos, 1, false);
			else
				pVector[j * row + i] = new Particle (pos, 1, true);
		}
	}
	
} // end make particles

/*--------------------------------------------------------------------
					Make Constraints
--------------------------------------------------------------------*/

void make_constraints(void){
	
	for(int ii = 0; ii < row; ii++){
		
		for(int jj = 0; jj < column; jj++){
						
			/* neighbors */
						
			if(ii < row-1) // to the right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+1, jj)) );
			
			if(jj < column -1) // below	
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii,jj+1) ) );
			
			if(ii < row-1 && jj < column -1) // down right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+1,jj+1) ) );
			
			if(ii < row-1 && jj < column -1) // up right	
				constraints.push_back( new Constraint(getParticle(ii+1,jj), getParticle(ii,jj+1) ) );
			
			/* neighbor's neighbors */
			
		 	if(ii < row-2) // to the right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+2, jj) ) );
			
			if(jj < column -2) // below	
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii,jj+2) ) );
			
			if(ii < row-2 && jj < column -2) // down right
				constraints.push_back(new Constraint(getParticle(ii,jj), getParticle(ii+2,jj+2) ) );
			
			if(ii < row-2 && jj < column -2) // up right	
				constraints.push_back( new Constraint(getParticle(ii+2,jj), getParticle(ii,jj+2) ) );
		}	
	}	
}

/*--------------------------------------------------------------------
					Update Normals
--------------------------------------------------------------------*/

void calculate_normals(){
	
	for(int ii = 0; ii < row-1; ii++){
		
		for(int jj = 0; jj < column-1; jj++){
			
			 // Particle 1 = ii+1, jj
			// Particle 2 = ii, jj
			// Particle 3 = ii, jj+1
			
			/* calculate normal of triangle */
			
			Vec3f normal = triangle_normal(getParticle(ii+1,jj)->m_Position, getParticle(ii,jj)->m_Position, getParticle(ii,jj+1)->m_Position);
			
			/* add component to triangle */
			
			getParticle(ii+1,jj)->updateNormal(normal); 
			getParticle(ii,jj)->updateNormal(normal);
		 	getParticle(ii+1,jj+1)->updateNormal(normal);
		
			/* calculate normal of triangle */
		
			normal = triangle_normal(getParticle(ii+1,jj+1)->m_Position, getParticle(ii+1,jj)->m_Position, getParticle(ii,jj+1)->m_Position);
		
			/* add component to triangle */
		
			getParticle(ii+1,jj+1)->updateNormal(normal);
			getParticle(ii+1,jj)->updateNormal(normal);
	 		getParticle(ii,jj+1)->updateNormal(normal);
		
		} // end for jj
	} // end for ii
	
}

/*--------------------------------------------------------------------
					Initialize System
--------------------------------------------------------------------*/

static void init_system(void)
{
	
	// Create three particles, attach them to each other, then add a
	// circular wire constraint to the first.
		pVector.resize(size);
		
		Vec3f sphere_center = Vec3(0.5f, -0.75f, 0.0f);
		float radius = 0.25f;
		
		shapes.push_back(new Sphere(radius, sphere_center, 0));
		
		make_particles(); // create particles
		make_constraints(); // create constraints
}

/*--------------------------------------------------------------------
					Draw Particles
--------------------------------------------------------------------*/

static void draw_particles ( void )
{
	int size = pVector.size();

	for(int ii=0; ii< size; ii++)
	{
		pVector[ii]->draw();
	}
}

/*--------------------------------------------------------------------
					Draw Foreces
--------------------------------------------------------------------*/
static void draw_forces ( void )
{}

/*----------------------------------------------------------------------
relates mouse movements to tinker toy construction
----------------------------------------------------------------------*/

static void remap_GUI()
{	
	int ii, size = pVector.size();
	for(ii=0; ii<size; ii++)
	{
		pVector[ii]->reset();
	}
}

/*----------------------------------------------------------------------
GLUT callback routines:: Keyboard and Mouse
----------------------------------------------------------------------*/

static void key_func ( unsigned char key, int x, int y )
{
	switch ( key )
	{
	case 'c':	
	case 'C':
		clear_data ();
		break;
	case ' ':
		dsim = !dsim;
		break;	
		
	case 'd': // move right
	case 'D':
		shapes[0]->m_position[0] += 0.1;
		break;
	case 'a': // move left
	case 'A':
		shapes[0]->m_position[0] -= 0.1;
		break;
	
	case 'w': // move up
	case 'W':
		shapes[0]->m_position[1] += 0.1;
		break;
	case 's': // move down
	case 'S':
		shapes[0]->m_position[1] -= 0.1;
		break;
	
	case 'z': // move in
		shapes[0]->m_position[2] -= 0.1;
		break;
	case 'Z': // move out
		shapes[0]->m_position[2] += 0.1;
		break;
	
	case 'v': // turn visible sphere on or off
	case 'V':
		visible = !visible;
		break;
	
	case 'q':
	case 'Q':
	case 27:
		free_data ();
		exit ( 0 );
		break;


	
	default:
		break;
	} // end switch

} // end keyboard

static void mouse_func ( int button, int state, int x, int y )
{}

static void motion_func ( int x, int y )
{}

static void reshape_func ( int width, int height )
{
	glutSetWindow ( win_id );
	glutReshapeWindow ( width, height );

	win_x = width;
	win_y = height;
}

static void step_func ( )
{
	if ( dsim )
		verlet_simulation_step(pVector, constraints, shapes[0]);
	
	else {remap_GUI();}

	glutSetWindow ( win_id );
		
}

static void display_func ( void )
{

	glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT );
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity() ;
	gluLookAt(x_camera, y_camera, z_camera,  // x,y,z coord of the camera 
			  lookAtX, lookAtY, lookAtZ,
			  0,1,0); // the direction of Up (default is y-axis)
	
	step_func();
	
	calculate_normals();
		
	draw_forces();
	draw_particles();

	if(visible)
		shapes[0]->draw();

	glutSwapBuffers ();
	
	glutPostRedisplay();
}

/*
----------------------------------------------------------------------
open_glut_window --- open a glut compatible window and set callbacks
----------------------------------------------------------------------
*/

static void open_glut_window ( void )
{
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );

	glutInitWindowPosition ( 0, 0 );
	glutInitWindowSize ( win_x, win_y );
	win_id = glutCreateWindow ( "Cloth!" );

	glEnable(GL_DEPTH_TEST);
	glClearColor(0,0,0,1);
	glMatrixMode(GL_PROJECTION) ;
	glLoadIdentity() ;
	gluPerspective(20, 1.0, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity() ;  // init modelview to identity
	
	glutKeyboardFunc ( key_func );
	glutMouseFunc ( mouse_func );
	glutMotionFunc ( motion_func );
	glutReshapeFunc ( reshape_func );
	glutDisplayFunc ( display_func );
}

/*----------------------------------------------------------------------
						main --- main routine
----------------------------------------------------------------------*/

int main ( int argc, char ** argv )
{
	glutInit ( &argc, argv );

	if ( argc == 1 ) {
		N = 64;
		d = 5.f;
		
	} else {
		N = atoi(argv[1]);
		d = atof(argv[3]);
	}
	
	init_system();
	
	win_x = 800;
	win_y = 800;
	open_glut_window ();

	glutMainLoop ();

	exit ( 0 );
}
