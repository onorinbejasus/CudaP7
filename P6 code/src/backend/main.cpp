
#include "lightfield.hpp"
#include "texture.hpp"
#include "common/pipeline/queue.hpp"
#include "common/network/packet.hpp"
#include "common/network/socket.hpp"
#include "backend/config.hpp"
#include "backend/stage/request_receiver.hpp"
#include "backend/stage/renderable.hpp"
#include "backend/stage/compressor.hpp"
#include "backend/stage/result_sender.hpp"
#include "backend/render/lightfield/lf.hpp"
#include <stdio.h>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <string>

static size_t QUEUE_SIZE = 16;
static size_t MAX_TRANSFER_SIZE = (1<<22);

using namespace lightfield;
using namespace aed::common::pipeline;
using namespace aed::common::util;
using namespace aed::backend;
using namespace aed::backend::stage;

struct InitData
{
	LFTextureSize ts;
	LFNumCameras nc; 
	const char* root;

	uint16_t port;
	size_t max_transfer;
	size_t max_frontends;
};

static int run_backend( const InitData& init_data )
{
	LFData lfdata;
	if ( !load_lightfield( &lfdata, init_data.root, init_data.nc, init_data.ts ) ) {
		return 1;
	}

	aed::common::network::SocketManager socket_manager;

    PipelineQueue<RenderableRequestData> render_input_queue;
    render_input_queue.initialize( QUEUE_SIZE );
    PipelineQueue<CompressorRequestData> compression_input_queue;
    compression_input_queue.initialize( QUEUE_SIZE );
    PipelineQueue<RSRequestData> sender_input_queue;
    sender_input_queue.initialize( QUEUE_SIZE );

    RequestReceiver rr( render_input_queue );
    socket_manager.initialize( &rr, init_data.port, MAX_TRANSFER_SIZE, init_data.max_frontends );
	Compressor compressor( sender_input_queue );
    ResultSender rs( &socket_manager, &rr );

	aed::backend::render::lightfield::LightfieldRenderAlgorithm lf( lfdata );
	Renderable render_alg( &lf, compression_input_queue, 0 );

    PipelineStage<Renderable, RenderableRequestData> render_stage( &render_alg, &render_input_queue );
    PipelineStage<Compressor, CompressorRequestData> compressor_stage( &compressor, &compression_input_queue );
    PipelineStage<ResultSender, RSRequestData> sender_stage( &rs, &sender_input_queue );
    
    // start threads
    boost::shared_ptr<boost::thread> renderable_thread( render_stage.run_on_new_thread() );
    boost::shared_ptr<boost::thread> compressor_thread( compressor_stage.run_on_new_thread() );
    boost::shared_ptr<boost::thread> sender_thread( sender_stage.run_on_new_thread() );
    boost::shared_ptr<boost::thread> receiver_thread( socket_manager.run() );
    
	std::string input;
    while (true) {
		std::cin >> input;
        static const std::string QUIT = "quit"; // do this, otherwise operator= allocates memory
        if (QUIT == input) {
            break;
        }
    }

    socket_manager.terminate();
    
    render_input_queue.abort_wait();
    compression_input_queue.abort_wait();
    sender_input_queue.abort_wait();

    receiver_thread->join();
    renderable_thread->join();
    compressor_thread->join();
    sender_thread->join();

	return 0;
}

static void print_usage( const char* progname )
{
    printf( "Usage: %s <port> <num_cameras> <image_size> [data_dir]\n\twhere\n\t"\
			"port = port to use for accepting requests\n\t"\
            "num_cameras = {545, 2113}\n\t"\
            "image_size = {128, 256}\n\t"\
            "data_dir is the root directory of the lightfield data,\n\t"\
            "or the default /afs location if not given. Do not provide\n\t"\
            "this argument when running on GHC.\n", progname );
}

int main(int argc, char** argv)
{
	InitData init_data;
    size_t num_cameras, image_size;
	int port;

    // start a new application
    if ( argc != 4 && argc != 5 ) {
        goto FAIL;
    }

    if ( 1 != sscanf( argv[1], "%d", &port ) || 1 != sscanf( argv[2], "%zu", &num_cameras ) || 1 != sscanf( argv[3], "%zu", &image_size ) ) {
        goto FAIL;
    }
    switch ( num_cameras ) {
    case 545:
        init_data.nc = LFNC_545;
        break;
    case 2113:
        init_data.nc = LFNC_2113;
        break;
    default:
        goto FAIL;
    }
    switch ( image_size ) {
    case 128:
        init_data.ts = LFTS_128;
        break;
    case 256:
        init_data.ts = LFTS_256;
        break;
    default:
        goto FAIL;
    }

    if ( argc == 5 ) {
        init_data.root = argv[4];
    } else {
        init_data.root = NULL;
    }

	init_data.port = port;
	init_data.max_frontends = 16;
	aed::log_init( "belog.txt", stdout );
	return run_backend( init_data );

  FAIL:
    print_usage( argv[0] );
    return 1;
}

