
#ifndef AED_FRONTEND_STAGE_RECEIVER_HPP
#define AED_FRONTEND_STAGE_RECEIVER_HPP

#include "common/network/socket.hpp"
#include "common/pipeline/pipeline.hpp"
#include "frontend/config.hpp"
#include <boost/thread.hpp>

namespace aed {
namespace frontend {
namespace stage {

typedef ResultData ReceiverOutput;

class Receiver : public common::network::IClient
{
public:
    Receiver(
        common::pipeline::PipelineQueue<ReceiverOutput>& queue
      )
      : queue( queue )
        {}
    virtual ~Receiver() {}
    virtual Result initialize( common::network::SocketManager* ) { return RV_OK; }
    virtual Result on_connection_closed( common::network::Socket* sock );
    virtual Result on_packet_received( common::network::Socket* sock, void* data, size_t len );

    boost::thread* run_on_new_thread();

    // HACK can't set this until SM is initialized
    common::network::Socket* socket;

private:

    common::pipeline::PipelineQueue<ReceiverOutput>& queue;
};

} // stage
} /* frontend */
} /* aed */

#endif /* AED_RECEIVER_HPP */

