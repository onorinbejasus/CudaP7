//
// client.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <vector>
#include "../Shared/connection.hh" // Must come before boost/serialization headers.
#include <boost/serialization/vector.hpp>

#include <cuda_runtime.h>

#include "../Shared/data.hh"
#include "window.hh"
#include "open_gl.hh"

#include <GLUT/glut.h>

#define min(a,b) ((a) < (b)? a:b)
#define PI 3.14159265
#define movingSpeed 10.0

GLuint display_list;		

// Camera variables
GLfloat posX, posY, posZ;
GLfloat cameraViewAngle, cameraSight;
GLfloat cameraSensitivity;

// Simulation variables
bool dsim = true;
bool wind = true;

namespace s11n_example {

/// Downloads stock quote information from a server.
class client
{
public:
  /// Constructor starts the asynchronous connect operation.
  client(boost::asio::io_service& io_service,
      const std::string& host, const std::string& service)
    : connection_(io_service)
  {
    // Resolve the host name into an IP address.
    boost::asio::ip::tcp::resolver resolver(io_service);
    boost::asio::ip::tcp::resolver::query query(host, service);
    boost::asio::ip::tcp::resolver::iterator endpoint_iterator =
      resolver.resolve(query);
    boost::asio::ip::tcp::endpoint endpoint = *endpoint_iterator;

    // Start an asynchronous connect operation.
    connection_.socket().async_connect(endpoint,
        boost::bind(&client::handle_connect, this,
          boost::asio::placeholders::error, ++endpoint_iterator));
    }

  /// Handle completion of a connect operation.
  void handle_connect(const boost::system::error_code& e,
      boost::asio::ip::tcp::resolver::iterator endpoint_iterator)
  {
    if (!e)
    {
      // Successfully established connection. Start operation to read the list
      // of vertices. The connection::async_read() function will automatically
      // decode the data that is read from the underlying socket.
      connection_.async_read(mesh_,
          boost::bind(&client::handle_read, this,
            boost::asio::placeholders::error));
    }
    else if (endpoint_iterator != boost::asio::ip::tcp::resolver::iterator())
    {
      // Try the next endpoint.
      connection_.socket().close();
      boost::asio::ip::tcp::endpoint endpoint = *endpoint_iterator;
      connection_.socket().async_connect(endpoint,
          boost::bind(&client::handle_connect, this,
            boost::asio::placeholders::error, ++endpoint_iterator));
    }
    else
    {
      // An error occurred. Log it and return. Since we are not starting a new
      // operation the io_service will run out of work to do and the client will
      // exit.
      std::cerr << e.message() << std::endl;
    }
  }

  /// Handle completion of a read operation.
  void handle_read(const boost::system::error_code& e)
  {
    if (!e)
    {
	  // initialize display list 
		
		display_list = glGenLists(1);
		glNewList(display_list, GL_COMPILE);
	
		glBegin(GL_TRIANGLES);
		
		for (std::size_t i = 0; i < mesh_.size(); ++i)
      	{
		
			glVertex3fv(mesh_[i].vertices[0]);
			glVertex3fv(mesh_[i].vertices[1]);
			glVertex3fv(mesh_[i].vertices[2]);
		
      	}
		glEnd();
		
		glEndList();
		
		std::cout << "done rendering. size: " << mesh_.size() << std::endl;
    }
    else
    {
      // An error occurred.
      std::cerr << e.message() << std::endl;
    }

    // Since we are not starting a new operation the io_service will run out of
    // work to do and the client will exit.
  }

private:
  /// The connection to the server.
  connection connection_;

  /// The data received from the server.
  std::vector<struct m_triangle> mesh_;
};

} // namespace s11n_example

// ============================
// = GLUT / OpenGL Call Backs =
// ============================

// ===========
// = Display =
// ===========

void display(void) {
    // Set the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 0.1, 100.0);

    // Set the modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Clear frame
    glClearColor(0.0f, 0.0f, 0.5f, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Set the camera
    gluLookAt(posX,
              posY,
              posZ,
              posX + cos(cameraViewAngle) * cos(cameraSight),
              posY + sin(cameraSight),
              posZ + sin(cameraViewAngle) * cos(cameraSight),
              cos(cameraViewAngle) * (-sin(cameraSight)),
              cos(cameraSight),
              sin(cameraViewAngle) * (-sin(cameraSight)));


    glColor3f(1.0, 0.0, 0.0);
  
    glCallList(display_list);
  
    /* buffer is ready */
    glutSwapBuffers();
	
    return;
}

// ===========
// = Reshape =
// ===========

void reshape(int width, int height) {
	glViewport(0, 0, width, height);
	
    // set the view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 0.1, 100.0);
	
	glutPostRedisplay();
}

// ===================================
// = GLUT and OpenGL setup functions =
// ===================================



/// The keyboard callback
void keyboard(
			  unsigned char key,	///< the key being pressed
			  int x,				///< x coordinate of the mouse
			  int y)				///< y coordinate of the mouse
{
	switch (key)
	{
        case 'j':
        case 'J':
            cameraViewAngle -= PI/cameraSensitivity;
            break;
        case 'l':
        case 'L':
            cameraViewAngle += PI/cameraSensitivity;
            break;
        case 'i':
        case 'I':
            cameraSight += 0.05;
            break;
        case 'k':
        case 'K':
            cameraSight -= 0.05;
            break;
        case 'a':
        case 'A':
            posX += movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            posZ -= movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            break;
        case 'd':
        case 'D':
            posX -= movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            posZ += movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            break;
        case 'w':
        case 'W':
            posX += movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            posZ += movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            break;
        case 's':
        case 'S':
            posX -= movingSpeed/cameraSensitivity*cos(cameraViewAngle);
            posZ -= movingSpeed/cameraSensitivity*sin(cameraViewAngle);
            break;
        case 'p':
        case 'P':
            printf("Camera: (%f, %f, %f,) at (%f, %f)\n", posX, posY, posZ, cameraViewAngle, cameraSight);
            break;
        case 'e':
        case 'E':
            posY -= 4.0/cameraSensitivity;
            break;
		case 'q':
		case 'Q':
            posY += 4.0/cameraSensitivity;
            break;
		case 27:
			// free_data();
			cudaThreadExit();
			exit(0);
			break;
			
		default:
			break;
	}
	
	glutPostRedisplay();
	
}

/// timer callback function
void my_timer(int id){
	
}

/// The mouse callback
void mouse(
		   int button, ///< which button was pressesd
		   int state,	///< up or down
		   int x,		///< x position
		   int y)		///< y position
{	
	return;
}

/// Mouse motion callback
void motion(
			int x,		///< x coordinate of mouse
			int y)		///< y coordinate of mouse
{
	return;
}


// ========
// = Main =
// ========

int main(int argc, char* argv[])
{
	createWindow(argc, argv);
	
	try
	{
	  // Check command line arguments.
	  if (argc != 3)
		{
		std::cerr << "Usage: client <host> <port>" << std::endl;
		return 1;
		}

	  boost::asio::io_service io_service;
	  s11n_example::client client(io_service, argv[1], argv[2]);
	  io_service.run();
	}
	catch (std::exception& e)
	{
	  std::cerr << e.what() << std::endl;
	}

	// Init the camera variables
	posX = 0.0; posY = 0.0; posZ = 0.0;
	cameraViewAngle = -1.5;
	cameraSight = 0.0;
	cameraSensitivity = 40;

	startApplication(argc, argv);

	return 0;
}
