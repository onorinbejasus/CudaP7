/** @file packet.hpp
 * 
 *  This file defines the structure, contain of the packets,
 *  including the request packet and the response packet. And 
 *  parsing/packetizing function for stream<->data conversion 
 *  between client/server 
 *
 */

#ifndef AED_PACKET_HPP
#define AED_PACKET_HPP

#include "common/math/camera.hpp"
#include "common/math/mouse.hpp"
#include <string.h>
#include <stdint.h>

namespace aed {

namespace common {
namespace network {
    struct Socket;
}
}

typedef uint64_t objid_t;

#define AED_MAX_BACKEND_STATE_LEN (1<<20)
#define AED_MAX_BACKEND_REQUEST_LEN ((1<<21) - 8)

struct TransformData
{
	Camera camera;
    uint32_t clip_width;
    uint32_t clip_height;
    uint32_t x_offset;
    uint32_t y_offset;
    uint32_t window_width;
    uint32_t window_height;
};

// Types of backend requests.
enum BackendRequestType
{
    // Render req from frontend node.
    BRT_RENDER,
    // State info from another (usually backend) node.
    BRT_STATE_DATA,
    // Input data from the frontend, such as mouse clicks
    BRT_FE_INPUT_DATA
};

// Header for every backend request.
struct BackendRequestHeader
{
    BackendRequestType type;    // request type
    objid_t renderable;         // destination renderable
    objid_t fe_ident;           // identifier given by frontend
};

// A full backend request packet plus source data
// Actual packets may only partially fill data.
// The contents/length of data are given by the header's type.
struct BackendRequestPacket
{
    common::network::Socket* source;
    size_t data_len;
    BackendRequestHeader header;
    uint8_t data[AED_MAX_BACKEND_REQUEST_LEN - sizeof(BackendRequestHeader)];
};

// Packet data for BRT_RENDER
struct BackendRenderingRequest
{
    // opaque identifier given by frontend, handed back unchanged
    //objid_t ident;
    // object/camera location
    TransformData transform;
};

// convenience struct for constructing rendering requests
struct BackendRenderingRequestPacket
{
    BackendRequestHeader header;
    BackendRenderingRequest request;
};


struct BackendInputNotification
{
    MouseData mouseData;
};

// struct for passing frontend interaction to the backend
struct BackendInputNotificationPacket
{
    BackendRequestHeader header;
    BackendInputNotification inputNotification;
};


// Packet data for BRT_STATE_DATA
struct BackendStateDataHeader
{
    objid_t type;
    objid_t stage;
    size_t len; // length of data, follows this byte
};

// convenience struct for constructing backend data packets with fixed size
template<typename T>
struct FixedLengthBackendStateDataPacket
{ 
    BackendRequestHeader packet_header;
    BackendStateDataHeader data_header;
    T data;

    // initializes everything excpept for the data
    FixedLengthBackendStateDataPacket( objid_t renderable, objid_t type )
    {
        packet_header.type = BRT_STATE_DATA;
        packet_header.renderable = renderable;
        data_header.type = type;
        data_header.len = sizeof(T);
    }
};

// convenience struct for constructing backend data packets with varying size
struct VariableLengthBackendStateDataPacket
{
    BackendRequestHeader packet_header;
    BackendStateDataHeader data_header;
    uint8_t data[AED_MAX_BACKEND_REQUEST_LEN - sizeof(BackendRequestHeader) - sizeof(BackendStateDataHeader)];

    // initializes everything
    VariableLengthBackendStateDataPacket( objid_t renderable, objid_t type, const void* data, size_t len )
    {
        packet_header.type = BRT_STATE_DATA;
        packet_header.renderable = renderable;
        data_header.type = type;
        data_header.len = len;
        memcpy( this->data, data, len );
    }

    // same, but for two buffers. the first 8 bytes will be the length of the first buffer, the second 8 the length of the second
    VariableLengthBackendStateDataPacket( objid_t renderable, objid_t type, const void* data1, size_t len1, const void* data2, size_t len2 )
    {
        packet_header.type = BRT_STATE_DATA;
        packet_header.renderable = renderable;
        data_header.type = type;
        data_header.len = len1 + len2 + 16;
        memcpy( (uint8_t*)this->data + 16, data1, len1 );
        memcpy( (uint8_t*)this->data + 16 + len1, data2, len2 );
        ((uint64_t*)this->data)[0] = len1;
        ((uint64_t*)this->data)[1] = len2;
    }

    size_t total_packet_len() const
    {
        return sizeof(BackendRequestHeader) + sizeof(BackendStateDataHeader) + data_header.len;
    }
};

enum BackendResponseDataType
{
    BRDT_COLOR_IMAGE,
    BRDT_DEPTH_IMAGE,
    BRDT_DATA,
};

struct BackendResponseHeader
{
    // identifier, as passed to backend
    objid_t fe_ident;
    // the type of data (e.g., color image, depth image)
    uint64_t data_type; 
    // length of data that follows immediately after packet header
	uint64_t data_length;
};

} /*aed*/

#endif /* AED_PACKET_HPP */

