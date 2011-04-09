

#ifndef AED_BACK_PLANE_REQUEST_RECEIVER_HPP
#define AED_BACK_PLANE_REQUEST_RECEIVER_HPP

#include "intermediate_result.hpp"
#include "common/network/socket.hpp"
#include "common/pipeline/queue.hpp"

#include <boost/pool/singleton_pool.hpp>
#include <boost/thread.hpp>


namespace aed {
namespace backend {
namespace stage {

// For lazy programmers like me. 
typedef BackendRequestPacket    RRResultData;

class RequestReceiver : public common::network::IServer
{
public:
    RequestReceiver( common::pipeline::PipelineQueue<RRResultData>& queue ) : queue( queue ) {}
    virtual ~RequestReceiver() {}
    virtual Result initialize( common::network::SocketManager* ) { return RV_OK; }
    virtual Result accept_connection( common::network::Socket* sock );
    virtual Result on_connection_closed( common::network::Socket* sock );
    virtual Result on_packet_received( common::network::Socket* sock, void* data, size_t len );

private:
    common::pipeline::PipelineQueue<RRResultData>& queue;

};

} // stage
} // namespace backend
} // namespace aed

#endif // AED_BACK_PLANE_REQUEST_RECEIVER_HPP


