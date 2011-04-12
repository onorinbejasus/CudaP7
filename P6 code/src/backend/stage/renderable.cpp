#include "renderable.hpp"
#include "frontend/opengl.hpp"
#include "common/util/imageio.hpp"
#include "common/log.hpp"
#include "common/util/timer.hpp"
#include "common/util/parse.hpp"
#include <iostream>
#include <sstream>

namespace aed {
namespace backend {
namespace stage {

static float DEFAULT_DT = 1.0f/30.0f;

static bool notSaved = true;

static void set_result_header (
        const BackendRenderingRequest& request,
        objid_t fe_ident,
        RenderableResultData& result,
        uint64_t data_type
    )
{
    if (BRDT_COLOR_IMAGE == data_type) {
        result.header.data_length = request.transform.clip_width * request.transform.clip_height * 4;
        result.header.data_type = BRDT_COLOR_IMAGE;
    } else {
        result.header.data_length = request.transform.clip_width * request.transform.clip_height;
        result.header.data_type = BRDT_DEPTH_IMAGE;
    }
    result.header.fe_ident = fe_ident;
}

// ----------------------------------------------------------------------------
// Renderable
// ----------------------------------------------------------------------------
Result Renderable::initialize()
{
    assert( render_algorithm );
    return render_algorithm->initialize( config_index );
}

void Renderable::finalize()
{
    render_algorithm->destroy();
}

Result Renderable::process( const BackendRequestPacket& packet )
{
    LOG_MSG(MDL_BACK_END, SVR_TRIVIAL, "+ Renderable::enter process().");

    bool is_update_done = false;

    switch ( packet.header.type ) {
        case BRT_STATE_DATA:
        {
            LOG_MSG(MDL_BACK_END, SVR_TRIVIAL, "Renderable got state data");

            assert( packet.data_len >= sizeof( BackendStateDataHeader ) );
            BackendStateDataHeader* request = (BackendStateDataHeader*) packet.data;
            assert( packet.data_len == (sizeof *request) + request->len );

            render::RenderableData rd;
            rd.source = packet.source;
            rd.type = request->type;
            rd.stage = request->stage;
            rd.data = ((uint8_t*)packet.data) + sizeof *request;
            rd.len = request->len;

            Result rv = render_algorithm->process_data( &is_update_done, rd );
            if ( rv_failed( rv ) ) {
                LOG_MSG( MDL_RENDER, SVR_ERROR, "IRenderAlgorithm::process_data failed!" );
                return rv;
            }
        }
        break;

        case BRT_RENDER:
        {
            /*
            ManagedTimer renderableTimer(
                    CONFIG_OBJ.innerRenderableHandle0, 
                    TIMER_MANAGER_OBJ.constructTrackedObjectId(0,0)
                    );*/

            LOG_MSG(MDL_BACK_END, SVR_TRIVIAL, "Renderable got a render request");
 
            // actually, do an update first, then only render after update phase
            Result rv = render_algorithm->begin_update( &is_update_done, DEFAULT_DT );
            
            if ( rv_failed( rv ) ) {
                LOG_MSG( MDL_RENDER, SVR_ERROR, "IRenderAlgorithm::begin_update failed!" );
                return rv;
            }

            // Send the backend the camera information
            BackendRenderingRequest* rr = (BackendRenderingRequest*) &(packet.data);
            render_algorithm->update_transform_data( rr->transform );

            request_source = packet.source;

            // store render data for later use (failing if an uncompleted request already exists)
            assert( packet.data_len == sizeof( BackendRenderingRequest ) );
            this->render_request = *((BackendRenderingRequest*) packet.data);
            this->is_rendering_request = true;
        }
        break;

        case BRT_FE_INPUT_DATA:
        {
            LOG_MSG(MDL_BACK_END, SVR_TRIVIAL, "Renderable got an input update");
            
            assert( packet.data_len == sizeof( BackendInputNotification ) );
            BackendInputNotification* inputPacket = (BackendInputNotification*) packet.data;

            // Update the backend state with the input data
            /*
            printf("Renderable got an input update: %f, %f, %f, %f, %d, %d\n", 
                    inputPacket->mouseData.srcx,
                    inputPacket->mouseData.srcy,
                    inputPacket->mouseData.relx,
                    inputPacket->mouseData.rely,
                    inputPacket->mouseData.buttonLeftDown,
                    inputPacket->mouseData.buttonRightDown
                    );
            */
            render_algorithm->update_input_data(inputPacket->mouseData);
        }
        break;
    }

    // compute result, if done
    if ( is_update_done ) {

        assert( is_rendering_request );
        is_rendering_request = false;

        switch ( render_algorithm->get_result_type() ) {
            case render::RRT_IMAGE:
            {
                LOG_VAR_MSG( MDL_RENDER, SVR_NORMAL, "Fulfilling image render request from %p", request_source );

                common::pipeline::QueuePusher<RenderableResultData> color_handle(queue);
                common::pipeline::QueuePusher<RenderableResultData> depth_handle(queue);
                RenderableResultData& color_result = *color_handle;
                RenderableResultData& depth_result = *depth_handle;

                set_result_header( render_request, packet.header.fe_ident, color_result, BRDT_COLOR_IMAGE );
                color_result.socket = request_source;
                set_result_header( render_request, packet.header.fe_ident, depth_result, BRDT_DEPTH_IMAGE );
                depth_result.socket = request_source;

                Result rv = render_algorithm->render( color_result.data, depth_result.data, render_request.transform );
                if ( rv_failed( rv ) ) {
                    LOG_MSG( MDL_RENDER, SVR_ERROR, "IRenderAlgorithm::render failed!" );
                    return rv;
                }

                /*
                if(notSaved) {
                    if(!imageio_save_image("backendResult.png",color_result.data,512,512))
                        printf("Couldn't save image\n");
                    else
                        notSaved = false;
                }
                */

                color_handle.push();
                depth_handle.push();
            }
            break;

            case render::RRT_DATA:
            {
                /*
                ManagedTimer renderableTimer(
                        CONFIG_OBJ.innerRenderableHandle1, 
                        TIMER_MANAGER_OBJ.constructTrackedObjectId(0,0)
                        );*/
               
                common::pipeline::QueuePusher<RenderableResultData> data_handle(queue);
                RenderableResultData& data_result = *data_handle;
                
                LOG_VAR_MSG( MDL_RENDER, SVR_NORMAL, "Fulfilling data render request from %p", data_result.socket );

                data_result.header.fe_ident = packet.header.fe_ident;
                data_result.header.data_type = BRDT_DATA;
                data_result.socket = request_source;

                size_t length;
                Result rv = render_algorithm->query_result( data_result.data, &length, sizeof data_result.data );
                data_result.header.data_length = length;

                if ( rv_failed( rv ) ) {
                    LOG_MSG( MDL_RENDER, SVR_ERROR, "IRenderAlgorithm::query_result failed!" );
                    return rv;
                }

                data_handle.push();
            }
            break;
        }
    }

    return RV_OK;
}

} // stage
} // namespace backend
} // namespace aed

