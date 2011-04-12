

#ifndef AED_BACK_PLANE_RESULT_SENDER_HPP
#define AED_BACK_PLANE_RESULT_SENDER_HPP

#include "intermediate_result.hpp"
#include "common/pipeline/queue.hpp"
#include "backend/stage/request_receiver.hpp"

namespace aed {
namespace backend {
namespace stage {

// For lazy programmers like me. 
// The prefix RS* is abbreviation of ResultSender
typedef IntermediateResult  RSRequestData;

class ResultSender
{
public:
    ResultSender( common::network::SocketManager* socket_manager, RequestReceiver* receiver )
      : socket_manager( socket_manager ), receiver( receiver ) {}

    Result initialize();
    void finalize();
    Result process( RSRequestData& request ) const;

private:
    common::network::SocketManager* socket_manager;
    RequestReceiver* receiver;
};


} // stage
} // namespace backend
} // namespace aed

#endif // AED_BACK_PLANE_RESULT_SENDER_HPP


