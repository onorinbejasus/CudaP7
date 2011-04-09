/** @file socket.hpp
 *
 *  This file defines the basic socket mechanics of the client and server,
 *  including the establishment of connection, sending and receiving data,
 *  closing connection and the architectures of client/server socket 
 *
 */

#ifndef AED_SOCKET_HPP
#define AED_SOCKET_HPP

#include "common/result.hpp"
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <errno.h>
#include <boost/thread.hpp>

const size_t MAXRECV = 10000u;
const size_t MAXSEND = 10000u;
const int MAXCONNECTIONS = 5;

namespace aed {
namespace common {
namespace network {

struct Socket;
class SocketManager;

class INetworkObject
{
public:
    virtual ~INetworkObject() {}
    virtual Result initialize( SocketManager* mgr ) = 0;
    virtual Result on_connection_closed( Socket* sock ) = 0;
    virtual Result on_packet_received( Socket* sock, void* data, size_t len ) = 0;
};

class IServer : public INetworkObject
{
public:
    virtual ~IServer() {}
    virtual Result accept_connection( Socket* sock ) = 0;
};

class IClient : public INetworkObject
{
public:
    virtual ~IClient() {}
};

class SocketManager
{
public:
    SocketManager();
    ~SocketManager();

    /// @remark. Throws exceptions for errors. Object is left in an invalid and undefined
    ///  state upon error (i.e., you can't try to initialize it again if it fails the first time).
    void initialize( IServer* server, uint16_t port, size_t max_packet_len, size_t max_connections );
    void initialize( IClient* client, size_t max_packet_len, size_t max_connections );

    boost::thread* run();
    void terminate();

    Result open_connection( Socket** sockrv, const char* host, uint16_t port );
    void close_connection( Socket* sock );
    /// XXX this function is NOT thread safe, only invoke from one thread
    /// (as in, you can call it on any thread, but don't call it from multiple threads)
    Result send_packet( Socket* sock, void* data, size_t len );

    struct Impl;

private:
    Impl* impl;

};

} // network
} // common
} // aed

#endif /* AED_SOCKET_HPP */

