#include "player.hpp"
#include "config.hpp"
#include <cassert>
#include <vector>
#include <iostream>

#ifdef AEDGL
#include <SDL/SDL.h>
#endif // AEDGL

namespace aed {
namespace frontend {


// XXX duplicated in texture.cpp
static unsigned char* read_entire_file( const char* input, size_t* lenp )
{
    FILE* file = fopen( input, "rb" );
    if ( !file ) {
        LOG_VAR_MSG(MDL_FRONT_END, SVR_ERROR, "Could not open file '%s'\n", input );
        return NULL;
    }

    // load entire file

    fseek( file, 0L, SEEK_END );
    size_t len = ftell( file );
    fseek( file, 0L, SEEK_SET );

    // XXX should this be tracked malloc?
    uint8_t* buffer = (uint8_t*) malloc( len );
    fread( buffer, 1, len, file );

    fclose( file );

    *lenp = len;
    return buffer;
}

void Player::activate_movie_mode( const char* filename, size_t start_frame )
{
    // load all the cameras from the file

    size_t len;
    uint8_t* buffer = read_entire_file( filename, &len );

    assert( len % sizeof( Camera ) == 0 && buffer);

    this->movie_script = (Camera*) buffer;
    this->movie_length = len / sizeof( Camera );
    this->current_movie_frame = start_frame;
}

void Player::process_cameras( const char* filename )
{
    size_t len;
    uint8_t* buffer = read_entire_file( filename, &len );

    assert( len % sizeof( Camera ) == 0 && buffer);

    size_t size = len / sizeof( Camera );

    std::vector<Camera> cams( size );
    memcpy( &cams[0], buffer, len );

    std::vector<Camera> cams2;

    // copy over non duplicate frames
    for ( size_t i = 0; i < cams.size(); i++ ) {
        if ( i == cams.size() - 1 || cams[i].position != cams[i+1].position ) {
            cams2.push_back( cams[i] );
        }
    }

    // smooth camera over nearby frames

#define SUMRANGE 6

    std::vector<Camera> cams3;
    for ( size_t i = 0; i < cams2.size(); i++ ) {
        if ( SUMRANGE < i && i < cams2.size() - SUMRANGE ) {
            Camera c = cams2[i];
            for ( size_t j = 1; j <= SUMRANGE; j++ ) {
                c.position += cams2[i-j].position;
                c.position += cams2[i+j].position;
            }
            c.position /= (2*SUMRANGE) + 1;
            cams3.push_back( c );
        } else {
            cams3.push_back( cams2[i] );
        }
    }

    FILE* f = fopen( filename, "wb" );
    fwrite( &cams3[0], sizeof cams3[0], cams3.size(), f );
    fclose( f );
}

void Player::run() {
	// the time interval between each rendered frame (ms)
#ifdef AEDGL
	int frame_interval = 1000 / feconfig.target_framerate;
    current_time = SDL_GetTicks();
    last_frame_time = current_time;
#endif // AEDGL

    LOG_MSG(MDL_FRONT_END, SVR_NORMAL, "Player::run... before enter loop");
	// main loop
    while (running) {
        // if quit was called, return
        if (!running)
            return;

        LOG_MSG(MDL_FRONT_END, SVR_NORMAL, "Player::run...");

        if (!feconfig.phonyMode) {
            // check for and handle events
            process_events();
        }

        // update state
        update();
	
#ifdef AEDGL
		current_time = SDL_GetTicks();
        
        LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "Before SDL_Delay in Player::run" );
        // constrain the framerate within the specified maximum value
        if( ( current_time - last_frame_time ) < frame_interval ) {
            SDL_Delay( frame_interval - ( current_time - last_frame_time ) );
			current_time = SDL_GetTicks();
        }
#endif // AEDGL
        LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "finish SDL_Delay in Player::run" );
	}

}

void Player::activate_capture_mode( const char* filename )
{
    this->camera_capture_file = fopen( filename, "wb" );
}

void Player::process_events() {

#ifdef AEDGL
    SDL_Event event;
    
    LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "Before peeping events" );
    int num_events = 0;
    while ( (num_events = SDL_PeepEvents(&event, 1, SDL_GETEVENT, SDL_ALLEVENTS)) > 0 ) {
        LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "finish peeping events" );
        
        if (-1 == num_events) {
	        LOG_VAR_MSG( MDL_FRONT_END, SVR_CRITICAL, 
	        			 "Player::process_events(): Found error when peeping events!!!" );
            break;
        }

        switch (event.type) {
            case SDL_QUIT:
                quit();
                break;

            case SDL_MOUSEMOTION:
                mouseData.srcx = ((double)event.motion.x) / feconfig.window_width;
                mouseData.srcy = ((double)event.motion.y) / feconfig.window_height;
                mouseData.relx = ((double)event.motion.xrel) / feconfig.window_width;
                mouseData.rely = ((double)event.motion.yrel) / feconfig.window_height;
                break;

            case SDL_MOUSEBUTTONDOWN:
                switch ( event.button.button ) {
                case SDL_BUTTON_LEFT:
                    mouseData.buttonLeftDown = 1;
                    break;
                case SDL_BUTTON_RIGHT:
                    mouseData.buttonRightDown = 1;
                    break;
                case SDL_BUTTON_MIDDLE:
                    mouseData.buttonMiddleDown = 1;
                    break;
                }
                break;

            case SDL_MOUSEBUTTONUP:
                switch ( event.button.button ) {
                case SDL_BUTTON_LEFT:
                    mouseData.buttonLeftDown = 0;
                    break;
                case SDL_BUTTON_RIGHT:
                    mouseData.buttonRightDown = 0;
                    break;
                case SDL_BUTTON_MIDDLE:
                    mouseData.buttonMiddleDown = 0;
                    break;
                }
                break;
            
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    quit();
                    break;
                case SDLK_c:
                    is_capturing_camera = !is_capturing_camera;
                    LOG_VAR_MSG(MDL_FRONT_END, SVR_CRITICAL, "new camera capture state: %s.\n", is_capturing_camera ? "ON" : "OFF" );
                    if ( camera_capture_file && !is_capturing_camera ) {
                        fclose( camera_capture_file );
                        camera_capture_file = NULL;
                    }
                    break;
                case SDLK_p:
                {
                    Camera cam;
                    camera_control.get_camera( cam );
                }
                    break;
                case SDLK_h:
                    _key_frame_curve.push_keyframe_info(camera_control.camera);
                    break;
                case SDLK_j:
                    _key_frame_curve.display_keyframe_info();
                    break;
                case SDLK_k:
                    _key_frame_curve.save_keyframe_info();
                    break;
                case SDLK_l:
                    _key_frame_curve.load_keyframe_info();
                    break;
                case SDLK_SPACE:
                    if (_key_frame_curve.IsReady()) {
                        _move_along_kfc = !_move_along_kfc;
                        if (_move_along_kfc)
                            camera_control.camera.position = _key_frame_curve.InitialCameraPosition();
                    } else {
                        LOG_MSG(MDL_FRONT_END, SVR_CRITICAL, "KeyFrameCurve is not ready, please load a file or record key frame now.");
                    }
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
        }
        camera_control.handle_event( event );
    }
#endif // AEDGL
    LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "finish processing events" );
}

void Player::update() {

    LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "Player::update");
    if (_move_along_kfc) {
        double dt = (current_time - last_frame_time);
        last_frame_time = current_time;

        Camera& camera_ref = camera_control.camera;
        _key_frame_curve.update_camera(camera_ref, dt);

        UserInputInfo uii;
        uii.camera = camera_ref;
        uii.mouseData = mouseData;

        common::pipeline::QueuePusher<PlayerOutput> handle( queue );
        *handle = uii;
        Vector3 pos = camera_ref.get_position();
        std::cout << camera_ref.position << camera_ref.orientation << std::endl;
//        LOG_VAR_MSG(MDL_FRONT_END, SVR_CRITICAL, "Camera position is (%f, %f, %f)", pos.x, pos.y, pos.z);
        handle.push();
    } else if ( movie_script == NULL ) {
        // normal mode
        double dt = (current_time - last_frame_time);
        last_frame_time = current_time;
        camera_control.update( dt );
        
        Camera last_camera = camera;
        camera_control.get_camera( camera );

        static size_t counter = 0;
        if ( camera_capture_file && is_capturing_camera && (counter++ % 3 == 0) ) {
            // capture mode
            fwrite( &camera, sizeof camera, 1, camera_capture_file );
            fflush( camera_capture_file );
        }

        // Send mouse and camera data
        UserInputInfo uii;
        uii.camera = camera;
        uii.mouseData = mouseData; 

        common::pipeline::QueuePusher<PlayerOutput> handle( queue );
        *handle = uii;
        handle.push();


    } else {

        int extension = 0;
        if ( current_movie_frame < movie_length+extension ) {
            // movie mode
            common::pipeline::QueuePusher<PlayerOutput> handle( queue );

            UserInputInfo uii;
            if(current_movie_frame < movie_length) {
                uii.camera = movie_script[current_movie_frame++];
                uii.camera.position.y += 1;
            }
            else {
                uii.camera = movie_script[movie_length-1];
                current_movie_frame++;
            }
            uii.mouseData = mouseData; 
            uii.camera.fov = (PI / 180.0) * CAMERA_FOV;
            
            *handle = uii;
            handle.push();
        }

        printf( "rendering frame %zu...\n", current_movie_frame-2 );

        if ( current_movie_frame == movie_length+extension ) {
            printf("Going to sleep...\n");
            sleep( 10 );
            exit( 0 );
        }
    }
}
	
} /* frontend */
} /* aed */

