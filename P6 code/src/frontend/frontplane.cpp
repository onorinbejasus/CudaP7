#include "frontplane.hpp"

#include "config.hpp"
#include "common/util/parse.hpp"
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#include "common/pipeline/pipeline.hpp"

#include "player.hpp"
#include "frontend/stage/analyzer.hpp"
#include "frontend/stage/receiver.hpp"
#include "frontend/stage/decompressor.hpp"
#include "frontend/stage/compositor.hpp"
#include "common/util/config_reader.hpp"

#ifdef AEDGL
#include <SDL/SDL.h>
#endif

namespace aed {
namespace frontend {

static Configuration nonc_feconfig;
const Configuration& feconfig = nonc_feconfig;

using namespace common::pipeline;
using namespace common::util;
using namespace frontend::stage;

static bool init_SDL() {
#ifdef AEDGL
    if ( SDL_Init( SDL_INIT_TIMER | SDL_INIT_VIDEO ) == -1 ) {
        std::cout << "Error initializing SDL: " << SDL_GetError() << std::endl;
        return false;
    }
#endif
    return true;
}

bool Frontplane::parse_config_files( char* fe_config_file, char* shared_config_file)
{
    // Load frontend config file
    ConfigReader fe_cr;
    if (!fe_cr.load_config_file(fe_config_file))
    {
        LOG_MSG( MDL_FRONT_END, SVR_CRITICAL, "Cannot load frontend config file.\n");
        return false;
    }
    const ConfigNode* fe_config = fe_cr.get_root_node()->get_node("frontend_settings");   
    
    // Load shared config file
    ConfigReader shared_cr;
    if (!shared_cr.load_config_file(shared_config_file))
    {
        LOG_MSG( MDL_FRONT_END, SVR_CRITICAL, "Cannot load shared config file.\n");
        return false;
    }
    const ConfigNode* shared_config = shared_cr.get_root_node()->get_node("shared_settings");

    // For multiple backends, need to store hostname and port number for each
    int num_backends = i32_from_str(shared_config->get_value("num_backends"));
    if ( num_backends <= 0 ) return false;
    nonc_feconfig.backends.resize( num_backends );
    for (int i = 0; i < num_backends; ++i) 
    {
        const ConfigNode* connection_info_node = shared_config->get_node("backend_connection_info", i);
        nonc_feconfig.backends[i].hostname = connection_info_node->get_value("hostname");
        nonc_feconfig.backends[i].port = (uint16_t) i32_from_str(connection_info_node->get_value("port_number"));
    }

    // Get display configuration
    nonc_feconfig.window_width  = i32_from_str(fe_config->get_value("window_width"));
    nonc_feconfig.window_height = i32_from_str(fe_config->get_value("window_height"));
    nonc_feconfig.phonyMode     = ("1" == fe_config->get_value("phony_mode")) ? true : false;

    // Get framerate settings
    nonc_feconfig.frames_in_flight = i32_from_str(fe_config->get_value("frames_in_flight"));
    if(nonc_feconfig.frames_in_flight > MAX_FRAMES_IN_FLIGHT) {
        LOG_VAR_MSG( MDL_FRONT_END, SVR_ERROR, "Frontend config wants %zu frames in flight but max is %d.", 
                nonc_feconfig.frames_in_flight, MAX_FRAMES_IN_FLIGHT);
        return false;
    }
    nonc_feconfig.target_framerate = i32_from_str(fe_config->get_value("target_framerate"));


    // Get fluid rendering settings
    nonc_feconfig.rendering_fluid   = i32_from_str(fe_config->get_value("rendering_fluid"));
    nonc_feconfig.enable_user_input = i32_from_str(fe_config->get_value("enable_user_input"));
 

    // Get models and textures
    nonc_feconfig.maple_shadow      = fe_config->get_value("maple_shadow");
    nonc_feconfig.chestnut_shadow   = fe_config->get_value("chestnut_shadow");
    nonc_feconfig.oak_shadow        = fe_config->get_value("oak_shadow");
    nonc_feconfig.plum_shadow       = fe_config->get_value("plum_shadow");
    nonc_feconfig.maple_dark_shadow = fe_config->get_value("maple_dark_shadow");
    nonc_feconfig.env_grid_model    = fe_config->get_value("env_grid_model");
    nonc_feconfig.env_shadow_plane_model = fe_config->get_value("env_shadow_plane_model");
    nonc_feconfig.env_grid_texture  = fe_config->get_value("env_grid_texture");
    nonc_feconfig.fluid_smoke_texture = fe_config->get_value("fluid_smoke_texture");

    // Get other configuration data
    nonc_feconfig.num_geometry       = fe_config->get_node_count("geometry_settings");
    nonc_feconfig.gather_timing_data = "1" == shared_config->get_value("gather_timing_data") ? true : false;
    nonc_feconfig.num_decompressors = i32_from_str(fe_config->get_value("num_decompressors"));

    
    // Initialize the geometry data
    for (int i = 0; i < nonc_feconfig.num_geometry; ++i)
    {
        geometries.push_back( Analyzer::GeometryData() );
        Analyzer::GeometryData& gd = geometries.back();
        
        const ConfigNode* gSettings = fe_config->get_node_num("geometry_settings",i);
        if (gSettings == NULL)
        {
            LOG_VAR_MSG( MDL_FRONT_END, SVR_ERROR, "Cannot find settings for geometry object %d.", i);
            return false;
        }

        int firstDelim, secondDelim;

        string translation_str = gSettings->get_value("translation");
        firstDelim = translation_str.find(",");
        secondDelim = translation_str.find(",", firstDelim+1);
        gd.transform.position = Vector3(
                f64_from_str(translation_str.substr(0,firstDelim)),
                f64_from_str(translation_str.substr(firstDelim+1, secondDelim - firstDelim - 1)),
                f64_from_str(translation_str.substr(secondDelim+1)));

        string bounding_box_str = gSettings->get_value("bounding_box");
        firstDelim = bounding_box_str.find(",");
        secondDelim = bounding_box_str.find(",", firstDelim+1);
        Vector3 bounding_box = Vector3(
                f64_from_str(bounding_box_str.substr(0,firstDelim)),
                f64_from_str(bounding_box_str.substr(firstDelim+1, secondDelim - firstDelim - 1)),
                f64_from_str(bounding_box_str.substr(secondDelim+1)));
        gd.bound.max = bounding_box / 2.0;
        gd.bound.min = -gd.bound.max;
        
        gd.transform.orientation = Quaternion::Identity;
        // = f64_from_str(gSettings->get_value("theta"));
        gd.backend.backend_id = i32_from_str(gSettings->get_value("backend_id"));
        gd.backend.type_id = i32_from_str(gSettings->get_value("type_id"));
        
        string type_name = gSettings->get_value("type_name");
        const ConfigNode* rSettings = shared_config->get_node("renderable_settings", type_name.c_str());
        if(rSettings == NULL) {
            LOG_VAR_MSG( MDL_FRONT_END, SVR_ERROR, "No renderable settings entry for type %s", type_name.c_str());
            return false;
        }

        gd.rdata.pass_num = i32_from_str(rSettings->get_value("pass_num"));
        gd.rdata.shadow_type = i32_from_str(rSettings->get_value("shadow_type"));
        gd.rdata.type = common::get_render_type(rSettings->get_value("render_alg_type").c_str());
    }
    return true;
}

bool Frontplane::initialize(char* fe_config_file, char* shared_config_file, int frontend_index, char* comment)
{
    // Parse config files
    if (!parse_config_files( fe_config_file, shared_config_file)) {
        return false;
    }

	// Initialize SDL
	if(!feconfig.phonyMode && !init_SDL()) {
		return false;
    }

    // Set our index
    nonc_feconfig.frontend_index = frontend_index;

    // Set the comment for timer logs
    TIMER_MANAGER_OBJ.setCommentString(comment);

	return true;
}

void Frontplane::activate_special_mode( FrontplaneMode mode, const char* filename, size_t data )
{
    this->mode = mode;
    this->mode_file = filename;
    this->mode_data = data;
}

void Frontplane::destroy()
{

}

void Frontplane::start( const char* hostname, uint16_t port )
{
    LOG_MSG( MDL_FRONT_END, SVR_INIT, "Starting frontplane" );

    //TIMER_MANAGER_OBJ.gatherTimingData = feconfig.gather_timing_data;
	nonc_feconfig.feMemoryHandle = TIMER_MANAGER_OBJ.registerMemTracker("All FE memory");

	nonc_feconfig.compositorTimerHandle = TIMER_MANAGER_OBJ.registerTimer("Compositor");
	nonc_feconfig.compositorObjRecTimerHandle = TIMER_MANAGER_OBJ.registerTimer("Compositor Obj Rec");
	nonc_feconfig.receiverTimerHandle = TIMER_MANAGER_OBJ.registerTimer("Receiver");
	nonc_feconfig.receiverTimerHandle2 = TIMER_MANAGER_OBJ.registerTimer("Receiver Post Handle");
	nonc_feconfig.analyzerTimerHandle = TIMER_MANAGER_OBJ.registerTimer("Analyzer");
	nonc_feconfig.analyzerObjectTimerHandle = TIMER_MANAGER_OBJ.registerTimer("Analyzer Per Object");
	nonc_feconfig.decompressorTimerHandle = TIMER_MANAGER_OBJ.registerTimer("Decompressor");

    MutexCvarPair frame_mcp;
    frame_mcp.is_aborted = false;

    size_t num_decompressors = feconfig.num_decompressors;

    PipelineQueue<AnalyzerInput> analyzer_input_queue;
    analyzer_input_queue.initialize( 1 );
    PipelineQueue<CompositorInput> compositor_input_queue;
    compositor_input_queue.initialize( 32 );
    PipelineQueue<DecompressorInput> decompressor_input_queue;
    decompressor_input_queue.initialize( 32 );

    LOG_MSG( MDL_FRONT_END, SVR_INIT, "queues created..." );

    Receiver receiver( decompressor_input_queue );
    // open connection
    // Allow as many socket connections as there are backends
    // XXX this is totally not a very good number to have here
    socket_manager.initialize( &receiver, feconfig.window_pixel_number() * 20 - 64, feconfig.backends.size() );

    nonc_feconfig.backends[0].hostname = hostname;
    nonc_feconfig.backends[0].port = port;


    Analyzer analyzer( &socket_manager, geometries, &frame_mcp, compositor_input_queue );
    Player player( analyzer_input_queue );
    Compositor compositor( &frame_mcp );
    Decompressor* decompressors[num_decompressors];
    for ( size_t i = 0; i < num_decompressors; ++i ) {
        decompressors[i] = new Decompressor( compositor_input_queue );
    }

    switch ( this->mode ) {
    case BPM_NORMAL: break;
    case BPM_CAPTURE:
        player.activate_capture_mode( this->mode_file );
        break;
    case BPM_MOVIE:
        player.activate_movie_mode( this->mode_file, this->mode_data );
        compositor.activate_save_images( this->mode_data );
        break;
    case BPM_PROCESS:
        player.process_cameras( this->mode_file );
        exit( 0 );
    }

    IPipelineStage<AnalyzerInput>* analyzer_stage = create_stage( &analyzer, &analyzer_input_queue );
    IPipelineStage<CompositorInput>* compositor_stage = create_stage( &compositor, &compositor_input_queue );
    IPipelineStage<DecompressorInput>* decompressor_stages[num_decompressors];
    for ( size_t i = 0; i < num_decompressors; ++i ) {
        decompressor_stages[i] = create_stage( decompressors[i], &decompressor_input_queue );
    }

    LOG_MSG( MDL_FRONT_END, SVR_INIT, "stages created, launching threads..." );
    
    // start threads
    boost::shared_ptr<boost::thread> analyzer_thread( analyzer_stage->run_on_new_thread() );
    boost::shared_ptr<boost::thread> compositor_thread( compositor_stage->run_on_new_thread() );
    boost::shared_ptr<boost::thread> recevier_thread( socket_manager.run() );
    boost::shared_ptr<boost::thread> decompressor_threads[num_decompressors];
    for ( size_t i = 0; i < num_decompressors; ++i ) {
        decompressor_threads[i] = boost::shared_ptr<boost::thread>( decompressor_stages[i]->run_on_new_thread() );
    }

    // run player until quit
    boost::thread player_thread( boost::bind( &Player::run, &player ) );

    player_thread.join();

    // Write timer data to file
    char timerDataFileName[50];
    snprintf(timerDataFileName, 50, "frontendTimerData%d.txt", feconfig.frontend_index);
    TIMER_MANAGER_OBJ.writeTimerDataToFile(timerDataFileName, false);
    
    // Write memory data to file
    char memDataFileName[50];
    snprintf(memDataFileName, 50, "frontendMemoryData%d.txt", feconfig.frontend_index);
    TIMER_MANAGER_OBJ.writeMemoryDataToFile(memDataFileName);

    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "joined player thread, aborting queues..." );

    // abort all threads
    boost::mutex::scoped_lock mcp_lock( frame_mcp.mutex );
    socket_manager.terminate();
    frame_mcp.is_aborted = true;
    frame_mcp.cvar.notify_all();
    mcp_lock.unlock();
    analyzer_input_queue.abort_wait();
    decompressor_input_queue.abort_wait();
    compositor_input_queue.abort_wait();
    socket_manager.terminate();

    LOG_MSG( MDL_FRONT_END, SVR_TRIVIAL, "queues aborted" );

    analyzer_thread->join();
    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "joined analyzer thread" );
    compositor_thread->join();
    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "joined compositor thread" );
    recevier_thread->join();
    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "joined receiver thread" );
    for ( size_t i = 0; i < num_decompressors; ++i ) {
        decompressor_threads[i]->join();
    }
    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "joined decompressor thread" );

    delete analyzer_stage;
    delete compositor_stage;
    //XXX delete decompressor_stage;
}
	
} /* frontend */
} /* aed */

