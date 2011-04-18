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
#include "../Shared/data.hh"

#include <GLUT/glut.h>

#define min(a,b) ((a) < (b)? a:b)

GLuint display_list;		


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

void my_display(void) {
  /* clear the buffer */
  /* NEW: now we have to clear depth as well */
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) ;

  glMatrixMode(GL_MODELVIEW) ;
  glLoadIdentity();
  gluLookAt(0.0, 5.0, 25.0,  // x,y,z coord of the camera 
	    0.0, 0.0, 0.0,  // x,y,z coord of the origin
	    0.0, 1.0, 0.0); // the direction of up (default is y-axis)  

  glColor3f(1.0, 0.0, 0.0);
  
  glCallList(display_list);
  
  glPopMatrix();
  /* buffer is ready */
  glutSwapBuffers();
	
  return ;
}

// ===========
// = Reshape =
// ===========
void my_reshape(int w, int h) {

  /* define viewport -- x, y, (origin is at lower left corner) width, height */
  glViewport (0, 0, min(w,h), min(w,h));
  return;
}

// ===================================
// = GLUT and OpenGL setup functions =
// ===================================

void glut_setup(void) {

  /* specify display mode -- here we ask for a double buffer and RGB coloring */
  /* NEW: tell display we care about depth now */
  glutInitDisplayMode (GLUT_DOUBLE |GLUT_RGB |GLUT_DEPTH);

  /* make a 400x400 window with the title of "Stitcher" placed at the top left corner */
  glutInitWindowSize(400,400);
  glutInitWindowPosition(0,0);
  glutCreateWindow("Mesh View 1.0");

  /*initialize callback functions */
  glutDisplayFunc( my_display );
  glutReshapeFunc( my_reshape ); 
 
  return ;
}

void gl_setup(void) {

  /* specifies a background color: black in this case */
  glClearColor(0,0,0,0) ;

  /* NEW: now we have to enable depth handling (z-buffer) */
  glEnable(GL_DEPTH_TEST);

  /* NEW: setup for 3d projection */
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity();
  // perspective view
  gluPerspective( 20.0, 1.0, 1.0, 100.0);
  return;
}

// ========
// = Main =
// ========

int main(int argc, char* argv[])
{
  glutInit( &argc, argv ) ;

  /* we define these setup procedures */
  glut_setup() ;  
  gl_setup() ;
	  
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

/* go into the main event loop */
  glutMainLoop() ;

  return 0;
}