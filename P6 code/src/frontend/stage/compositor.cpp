#include "compositor.hpp"
#include "common/util/parse.hpp"
#include "common/util/imageio.hpp"
#include "frontend/render/image_save.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

#ifdef AEDGL
#include <SDL/SDL.h>
#endif


namespace aed {
namespace frontend {
namespace stage {

using namespace common::util;
using namespace frontend::render;

#ifdef AEDGL
static bool init_SDL_window( int width, int height, const char* title );
static bool initialize_OpenGL();
static void deinit_OpenGL();
static void display_FPS( size_t& frame_count, Timer &FPS_timer );

#endif // AEDGL

Result Compositor::initialize()
{
#ifdef AEDGL
    if (!feconfig.phonyMode) {

        // Init the SDL window
        if( !init_SDL_window( feconfig.window_width, feconfig.window_height, "Frontend") ||
            !initialize_OpenGL() ) 
        {
            LOG_MSG(MDL_FRONT_END, SVR_INIT, "SDL window or OpenGL is not initialized properly.");
            return RV_RESOURCE_UNAVAILABLE;
        }

        if(feconfig.rendering_fluid)
        {
        }

        lfrender.initialize();

        if(feconfig.rendering_fluid)
        {
        }

        LOG_MSG(MDL_FRONT_END, SVR_INIT, "Compositor: Window initialized ");


        //environment.initialize();


    } else { // PHONY MODE
        LOG_MSG(MDL_FRONT_END, SVR_INIT, "running in PHONY mode.");
    }
#endif // AEDGL
        
	// clear table
	for ( size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ ) {
        memset( &frame_data[i].received_table, 0, sizeof frame_data[i].received_table );
        frame_data[i].info = NULL;
	}

    storage_pool = new common::util::FixedPool( MAX_OBJECTS * 4, sizeof( ResultData ) );

    LOG_MSG(MDL_FRONT_END, SVR_INIT, "Compositor: Initialization complete.");

    return RV_OK;
}

void Compositor::finalize()
{
#ifdef AEDGL
    if (!feconfig.phonyMode) {
        deinit_OpenGL();
    }
#endif // AEDGL
}

void Compositor::activate_save_images( size_t start_image )
{
    do_save_images = true;
    image_counter = start_image;
}

Result Compositor::process( const CompositorInput& request )
{
     LOG_MSG(MDL_FRONT_END, SVR_NORMAL, "compositor::process...");

#ifdef AEDGL
    if (!feconfig.phonyMode) {
        // events MUST be pumped on this thread
        // just do it once per loop and hope this is often enough
        SDL_PumpEvents();
    }
#endif // AEDGL

    switch ( request.type )
    {
    case CompositorInput::CI_FrameInfo:
        // if frame info, update next frame data
        process_frame_info( request.frame_info );
        break;
    case CompositorInput::CI_ResultData:
        process_object( &request.result_data );
        break;
    }

    // check if all objects for current frame have been received, if so, render
    check_and_render();

    return RV_OK;
}

Result Compositor::process_frame_info( FrameInfo* info )
{
    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "Compositor received frame info" );
    // make sure we aren't receiving a duplicate
    assert( info && !frame_data[info->frame].info );

    frame_data[info->frame].info = info;
    // make sure we didn't get too many objects
    assert( info->objects.size() <= MAX_OBJECTS );
    assert( !info->is_rendered );

    // DON'T reset table, since we may have already received images for this frame

    // only process data if the info is for the current frame (otherwise we
    // would have processed it when we incremented the current frame
    if ( info->frame == current_frame ) {
        begin_frame( &frame_data[info->frame] );
    } else {
        assert( frame_data[current_frame].early_data.empty() );
    }

    return RV_OK;
}

Result Compositor::process_object( const ResultData* data )
{
    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "Compositor received object info" );

    ManagedTimer compositorObjectTimer(
            feconfig.compositorObjRecTimerHandle, 
            data->header.fe_ident);

    assert( data );

    uint64_t type = data->header.data_type;
    if ( type != BRDT_COLOR_IMAGE && type != BRDT_DEPTH_IMAGE && type != BRDT_DATA ) {
        LOG_VAR_MSG( MDL_RENDER, SVR_ERROR, "Received object with invalid type %u", (unsigned) data->header.data_type );
        return RV_INVALID_ARGUMENT;
    }

    // Extract frame index and object index
    // |--Frontend id--|---Frame num---|----Object z-sort index----|
    // |    16 bits    |    16 bits    |          32 bits          |
    size_t frame = 0x0000ffff & (data->header.fe_ident >> 32);
    size_t index = 0xffffffff & (data->header.fe_ident >>  0);

    // check for valid bounds
    if ( frame >= feconfig.frames_in_flight ) {
        LOG_VAR_MSG( MDL_RENDER, SVR_ERROR, "Received object with invalid frame %zu", frame );
        return RV_INVALID_ARGUMENT;
    }
    if ( index >= MAX_OBJECTS ) {
        LOG_VAR_MSG( MDL_RENDER, SVR_ERROR, "Received object with invalid index %zu", index );
        return RV_INVALID_ARGUMENT;
    }
    // check for duplicates
    if ( frame_data[frame].received_table[index][type == BRDT_DATA ? 0 : type] ) {
        LOG_VAR_MSG( MDL_RENDER, SVR_ERROR, "Received object with duplicate index %zu, type %zu, frame %zu", index, type, frame );
        return RV_INVALID_ARGUMENT;
    }

    // if early, add it to the early list, if possible
    if ( frame != current_frame || frame_data[frame].info == NULL ) {
        LOG_MSG( MDL_RENDER, SVR_DEBUG2, "Received object for future frame, attemping to store." );
        if ( storage_pool->capacity() > storage_pool->size() ) {
            ResultData* rd = (ResultData*) storage_pool->allocate();
            assert( rd );
            rd->header = data->header;
            memcpy( rd->data, data->data, data->header.data_length );
            frame_data[frame].early_data.push_back( rd );
            return RV_OK;
        }
    }

    // mark received
    switch ( type ) {
    case BRDT_DATA:
        // data counts for both types
        frame_data[frame].received_table[index][0] = true;
        frame_data[frame].received_table[index][1] = true;
        break;
    default:
        // else just mark correct entry
        frame_data[frame].received_table[index][type] = true;
    }
    LOG_VAR_MSG( MDL_RENDER, SVR_TRIVIAL, "Received object for frame %zu with index %zu", frame, index );

    // if we reach this, it's because we're discarding the data
    if ( frame != current_frame || frame_data[frame].info == NULL ) {
        LOG_MSG( MDL_RENDER, SVR_WARNING, "Out of memory for storing objects, discarding." );
        return RV_OK;
    }

    FrameInfo* info = frame_data[frame].info;

    // check to make sure this object is valid for this frame
    if ( index >= info->objects.size() ) {
        LOG_VAR_MSG( MDL_RENDER, SVR_ERROR, "Received object with no index in frame, index=%zu", index );
        return RV_INVALID_ARGUMENT;
    }

#ifdef AEDGL
    if (!feconfig.phonyMode) {
        // load into texture
        if (BRDT_DATA == type) {

            // XXX BUG index is not what this code believes:
            // index is the index is the CURRENT FRAME's object list,
            // but this code interprets it as the GEOMETRY ID, which is
            // totally not correct at all, and only happens for work in a few
            // special cases.

            LOG_VAR_MSG( MDL_RENDER, SVR_NORMAL, "Frontend received %zu bytes from backend", data->header.data_length );
            LOG_VAR_MSG( MDL_RENDER, SVR_NORMAL, "Updating frontend info for object index %zu", index );


        } else {
            lfrender.add_texture( (BackendResponseDataType)type, index, info->objects[index].clip, *data );  
        }
    }
#endif // AEDGL

    return RV_OK;
}

Result Compositor::begin_frame( CompositorFrameData* data )
{
    assert( data && data->info && &frame_data[current_frame] == data );

    // notify all render algorithms that the frame has started
#ifdef AEDGL
    if (!feconfig.phonyMode) {
        lfrender.begin_frame( data->info );
    }
#endif

    // process all early data for the next frame
    for ( size_t i = 0; i < data->early_data.size(); i++ ) {
        Result rv = process_object( data->early_data[i] );
        storage_pool->free( data->early_data[i] );
        if ( rv_failed( rv ) ) {
            return rv;
        }
    }
    data->early_data.clear();

    return RV_OK;
}

static bool are_all_received( const CompositorFrameData& data )
{
    // XXX this is dumb, replace with a counter so it's O(1) instead of O(n)

    if ( !data.info ) {
        LOG_MSG(MDL_FRONT_END, SVR_ERROR, "CompositorFrameData does not has info structure.");
        return false;
    }

    size_t num_objects = data.info->objects.size();
    for ( size_t i = 0; i < num_objects; ++i ) {
        if ( !data.received_table[i].received_all() ) {
            return false;
        }
    }

    return true;
}


#ifdef AEDGL
void Compositor::render( const CompositorFrameData& data )
{
    // Object id has no meaning here so just write the frame number
    ManagedTimer compositorTimer(
            feconfig.compositorTimerHandle,
            MAKE_TIMER_ID(0,data.info->frame,feconfig.frontend_index)); 
    
    static size_t frame_count = 0;
    static Timer FPS_timer;

    assert( data.info );

    const size_t num_objects = data.info->objects.size();

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
    glEnable( GL_DEPTH_TEST );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );

    // Clear the buffer with sky blue
    glClearColor( 74.0/255.0, 139.0/255.0, 244.0/255.0, 1.0 );
    //glClearDepth( 1.0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    
    // First pass: Draw sky and ground plane
    //environment.render_env(data.info->camera);

    // Clear depth buffer so nothing zfights with ground plane
    glClear( GL_DEPTH_BUFFER_BIT );

    /////////////////////////////////////////////////////////////
    // XXX Make sure the grass_id in the frag shader matches the
    // type id of the grass in the config. That way, the grass is
    // rendered under the shadow in the frontend.
    /////////////////////////////////////////////////////////////

    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();

    // Second pass: Draw grass 
    lfrender.render_pass( 1 );
    
    // Third pass: Add shadows to ground plane
    glDisable(GL_DEPTH_TEST);
    //environment.render_shadow(data.info);
    glEnable(GL_DEPTH_TEST);

    // Fifth pass: Render other objects (tress, etc.)
    lfrender.render_pass( 2 );
    
    glMatrixMode( GL_PROJECTION );
    glPopMatrix();
    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();
    
    // swap buffers
    SDL_GL_SwapBuffers();

    if( ++frame_count==30 ) {
        display_FPS( frame_count, FPS_timer );
    }

    lfrender.end_frame();
}
#endif // AEDGL


Result Compositor::check_and_render()
{
    CompositorFrameData& data = frame_data[current_frame];

    // check if we've received all objects
    if ( are_all_received( data ) ) {
#ifdef AEDGL
        if (!feconfig.phonyMode) {

            if(FLUID_DIM == 2) {
                static size_t frame_count = 0;
                static Timer FPS_timer;
 
                if( ++frame_count==25 ) {
                    display_FPS( frame_count, FPS_timer );
                }
            }
            else {
                // render
                render( data );
            }
        }
#endif // AEDGL

        // mark frame complete, notify analzyer
        //LOG_VAR_MSG( MDL_RENDER, SVR_TRIVIAL, "Frame %zu Complete", current_frame );
        boost::mutex::scoped_lock lock( frame_mcp->mutex );
        data.info->is_rendered = true;
        frame_mcp->cvar.notify_one();
        lock.unlock();

        LOG_VAR_MSG( MDL_RENDER, SVR_TRIVIAL, "Frame %zu rendered, clearing table", current_frame );

        // clear out data
        memset( &data.received_table, 0, sizeof data.received_table );
        data.info = NULL;
        assert( data.early_data.empty() );
        // increment frame
        current_frame++;
        current_frame %= feconfig.frames_in_flight;

        // if we've already received the frame info for the next frame, process all data
        if ( frame_data[current_frame].info != NULL ) {
            begin_frame( &frame_data[current_frame] );
        }

        // save image
#ifdef AEDGL
        if ( do_save_images ) {
            char filename[512];
            snprintf( filename, sizeof filename, "shots/shot%04zu.png", image_counter++ );
            imageio_save_screenshot( filename, feconfig.window_width, feconfig.window_height );
        }
#endif // AEDGL
    }
    
    return RV_OK;
}

#ifdef AEDGL
static bool init_SDL_window( int width, int height, const char* title ) {
    SDL_WM_SetCaption( title, title );

    // used the preferred bpp
    unsigned int bits_per_pixel = SDL_GetVideoInfo()->vfmt->BitsPerPixel;
    unsigned int flags = SDL_OPENGL;

    // set up some SDL related OpenGL stuff
    SDL_GL_SetAttribute( SDL_GL_BUFFER_SIZE, bits_per_pixel );
    SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );

    // create the window
    if ( SDL_SetVideoMode( width, height, bits_per_pixel, flags ) == 0 ) {
        LOG_VAR_MSG(MDL_FRONT_END, SVR_INIT, "Error initializing SDL surface: %s, aborting initialization.", SDL_GetError());
        return false;
    }

    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
    glPixelStorei( GL_PACK_ALIGNMENT, 1 );

    return true;
}


static bool initialize_OpenGL() {
    // initialize GLEW
    GLenum error = glewInit();
    if ( error != GLEW_OK ) {
        LOG_MSG(MDL_FRONT_END, SVR_INIT, "GLEW failed to initialize.");
        return false;
    }


    // set up basic GL state
    // XXX WHAT IS THIS YOU GUYS JUST COPIED SHIT FROM OTHER CODE AND DONT KNOW WHAT IT DOES ARL:KSFJD:JKLSGDJKL::: SFD
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);


    return true;
}

// HACK
static size_t num_updates = 0;
static double avg_fps = 0.0;

static void display_FPS( size_t& frame_count, Timer &FPS_timer ) {

    char title[100];
    double fps = ((double)frame_count)/FPS_timer.getElapsedTimeInSec();

	//LOG_VAR_MSG(MDL_FRONT_END, SVR_NORMAL, "frame count = %f", frame_count);
	//LOG_VAR_MSG(MDL_FRONT_END, SVR_NORMAL, "time = %f", FPS_timer.getElapsedTimeInSec());
    sprintf(title, "%f fps", (float)fps );
    SDL_WM_SetCaption(title, NULL);
    frame_count = 0;
	FPS_timer.start();

    // this will intentionally skip the first fps so things can settle
    if ( num_updates != 0 ) {
        avg_fps *= (num_updates-1);
        avg_fps += fps;
        avg_fps /= num_updates;
    }

    num_updates++;
}

static void deinit_OpenGL() {
    printf( "TOTAL FPS: %f\n", (float)avg_fps );
}	
#endif // AEDGL


} // stage
} /* frontend */
} /* aed */

