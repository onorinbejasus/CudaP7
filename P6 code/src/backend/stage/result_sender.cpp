#include <iostream>
#include "result_sender.hpp"
#include "common/util/imageio.hpp"

namespace aed {
namespace backend {
namespace stage {


Result ResultSender::initialize()
{
    return RV_OK;
}

void ResultSender::finalize() { }

Result ResultSender::process( RSRequestData& request ) const
{
    common::network::Socket* destination = request.socket;
    size_t len = (sizeof request.header) + request.header.data_length;
    Result rv = socket_manager->send_packet( destination, &request, len );

    LOG_VAR_MSG( MDL_BACK_END, SVR_NORMAL, "Sent packet with backend id %zu, backend_type %zu, lenth %zu",
         request.header.fe_ident, request.header.data_type, len );
    
    if ( rv_failed( rv )) {
        LOG_MSG(MDL_BACK_END, SVR_CRITICAL, "RS2::Cannot send request.");
        // don't return error code, keep running
    }

    return RV_OK;
}

} // stage
} // namespace backend
} // namespace aed

