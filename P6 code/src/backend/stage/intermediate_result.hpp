#ifndef AED_BACK_PLANE_INTERMEDIATE_RESULT_HPP
#define AED_BACK_PLANE_INTERMEDIATE_RESULT_HPP

#include "common/network/packet.hpp"
#include "common/network/socket.hpp"
#include "backend/config.hpp"
#include <utility>

namespace aed {
namespace backend {
namespace stage {

struct IntermediateResult
{
    // in correct ordering for packet so no copying necessary
    BackendResponseHeader header;
    uint8_t data[MAX_BUFFER_SIZE];
    common::network::Socket* socket;
    // XXX dimension?

    // we need this because otherwise, the default ctor takes 10 ms to run (?)
    IntermediateResult() { }
};


} // stage
} // namespace backend
} // namespace aed

#endif // AED_BACK_PLANE_INTERMEDIATE_RESULT_HPP

