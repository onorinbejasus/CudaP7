/**
 *	@file	 player.hpp
 *  @brief	 This class is the frontest working module, it take the input message from keyboard/mouse,
 *			 process the input event, and put the camera info into the request queue of analyzer,
 *			 so the render result could be present to the user, the class will keep looping until user
 *			 terminate it
 *  @author  Wei-Feng Huang
 *	@date	 July 8, 2010    	
 */

#ifndef AED_PLAYER_HPP
#define AED_PLAYER_HPP
#include "common/math/keyframecurve.hpp"
#include "common/math/camera_roam.hpp"
#include "common/math/mouse.hpp"
#include "common/pipeline/queue.hpp"
#include <cstdio>

namespace aed {
namespace frontend {

// TODO  player actually got no request data in the queue, its request come from keyboard and mouse
//		 the result data is camerainfo (we should set up a structure for that)

struct UserInputInfo {
    Camera camera;
    MouseData mouseData;
};

typedef UserInputInfo PlayerOutput;

class Player
{
public:
    Player( common::pipeline::PipelineQueue<PlayerOutput>& queue )
        : running( true ), queue(queue),
          current_time( 0 ), last_frame_time( 0 ),
          movie_script( 0 ), movie_length( 0 ), current_movie_frame( 0 ),
          is_capturing_camera( false ), camera_capture_file( 0 ),
          _move_along_kfc( false )
        {}

    ~Player() {}

    void activate_movie_mode( const char* filename, size_t start_frame );
    void activate_capture_mode( const char* filename );

    void process_cameras( const char* filename );

    void run();

	bool running;

private:
	void process_events();
	void update();
	inline void quit() {
        running = false;
        if ( camera_capture_file ) { fclose( camera_capture_file ); }
    }

    common::pipeline::PipelineQueue<PlayerOutput>& queue;

	int current_time;
	int last_frame_time;
	CameraRoamControl camera_control;

	// TODO this can be local variable, now for the comparison with camera_info last frame, 
	//		leave it as class member
	Camera camera;

    // Mouse input data to be sent to analyzer
    MouseData mouseData;

    // list of cameras for each frame of movie mode
    Camera* movie_script;
    // number of cameras
    size_t movie_length;
    size_t current_movie_frame;

    bool is_capturing_camera;
    FILE* camera_capture_file;

    KeyFrameCurve _key_frame_curve;
    bool _move_along_kfc;
};

} /* frontend */
} /* aed */
#endif /* AED_PLAYER_HPP */
