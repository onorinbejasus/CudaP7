
#include "receiver.hpp"
#include "common/log.hpp"
#include <boost/bind.hpp>

namespace aed {
namespace frontend {
namespace stage {

Result Receiver::on_connection_closed( common::network::Socket* )
{
    LOG_MSG( MDL_FRONT_END, SVR_FATAL, "backend disconnected, aborting" );
    abort();
}

Result Receiver::on_packet_received( common::network::Socket*, void* packet_data, size_t /* XXX should use for validation len*/ )
{
    BackendResponseHeader* header = (BackendResponseHeader*) packet_data;

    common::util::ManagedTimer receiverTimer(
            feconfig.receiverTimerHandle, 
            header->fe_ident); 

    // XXX TODO need to validate data

    common::pipeline::QueuePusher<ReceiverOutput> handle( queue );

    common::util::ManagedTimer receiverTimer2(
            feconfig.receiverTimerHandle2, 
            header->fe_ident); 

    ResultData& rd = *handle;
    rd.header = *header;

    memcpy( &rd.data, ((uint8_t*)packet_data) + sizeof *header, header->data_length );
    LOG_VAR_MSG( MDL_FRONT_END, SVR_TRIVIAL, "received new header: ident=%llu, length=%llu", (unsigned long long) header->fe_ident, (unsigned long long) header->data_length );
    handle.push();

    return RV_OK;
}

} // stage
} /* frontend */
} /* aed */

