#ifndef WINDOW_HH
#define WINDOW_HH

/// Creates a window.
void createWindow(
	int argc,       ///< number of arguments
	char ** argv);  ///< list of arguments

/// Starts the application. Must be called after createWindow().
void startApplication(
	int argc,       ///< number of arguments
	char ** argv);  ///< list of arguments
	
#endif /* end of include guard: WINDOW_HH */
