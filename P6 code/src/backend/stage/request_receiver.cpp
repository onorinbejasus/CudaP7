// TODO:
// 1. right now we just put request to queue, but if request is much faster 
//    processing, the queue will be exploded very quickly. We need a buffer 
//    to refresh previous un-processed request.
//------------------------------------------------------------------------

#include "request_receiver.hpp"
#include "common/log.hpp"
#include <boost/bind.hpp>

namespace aed {
namespace backend {
namespace stage {

Result RequestReceiver::accept_connection( common::network::Socket* )
{
    return RV_OK;
}

Result RequestReceiver::on_connection_closed( common::network::Socket* )
{
    return RV_OK;
}

Result RequestReceiver::on_packet_received( common::network::Socket* sock, void* data, size_t len )
{
    const BackendRequestHeader* header = (BackendRequestHeader*) data;
    const void* body = ((uint8_t*) data) + sizeof *header;

    if ( len < sizeof( BackendRequestHeader ) ) {
        return RV_INVALID_ARGUMENT;
    }

    LOG_VAR_MSG( MDL_NETWORK, SVR_NORMAL, 
            "Received packet from %p of size %zu and type %d to renderable %d", 
            sock, len, (int)header->type, (int)header->renderable );

    switch ( header->type ) {
        case BRT_RENDER:
            //printf("It's render data\n");
            break;
        case BRT_STATE_DATA:
            break;
        case BRT_FE_INPUT_DATA:
            //printf("It's input data\n");
            break;
        default:
            return RV_INVALID_ARGUMENT;
    }

    common::pipeline::QueuePusher<BackendRequestPacket> handle( queue );
    BackendRequestPacket& result = *handle;

    result.source = sock;
    result.data_len = len - sizeof *header;
    result.header = *header;
    memcpy( result.data, body, result.data_len );

    handle.push();

    return RV_OK;
}

} // stage
} // namespace backend
} // namespace aed

