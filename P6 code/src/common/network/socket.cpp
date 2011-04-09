#include "common/network/socket.hpp"
#include "common/log.hpp"
#include "common/util/fixed_pool.hpp"

#include <vector>
#include <string>
#include <errno.h>
#include <iostream>
#include <cstring>

#include <fcntl.h>
#include <netinet/tcp.h>

#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#include <tr1/unordered_map>

namespace aed {
namespace common {
namespace network {

//////////////////
// TYPES
//////////////////


enum NodeType { SM_CLIENT, SM_SERVER };

// struct Socket it not an actual type, just a handle used for the map.

struct SocketData
{
    uint8_t* buffer;
    sockaddr_in addr;
    int fd; 
    size_t bytes_buffered;

    SocketData() : buffer(0), fd(0), bytes_buffered(0) { }
};

typedef std::tr1::unordered_map<Socket*, SocketData> SocketMap;

struct SocketManager::Impl
{
    // constant information
    size_t max_packet_len;
    size_t max_connections;
    size_t buffer_size; // diff than max_packet_len

    // the handler and related data
    NodeType type;    
    union {
        IClient* client;
        IServer* server;
    };
    SocketData my_sock;

    // active sockets
    SocketMap sockets;
    // next socket number to use
    size_t socket_counter;
    // pool for socket buffers
	common::util::FixedPool* pool;

    // flag for if we're in run loop
    bool is_running;
    bool is_initialized;
    boost::mutex state_mutex;

    bool check_is_running()
    {
        boost::mutex::scoped_lock lock( state_mutex );
        return is_running;
    }

    INetworkObject* get_network_object()
    {
        switch ( type ) {
        case SM_SERVER: return server;
        case SM_CLIENT: return client;
        default: return NULL;
        }
    }

    Impl() : client(0), is_running(false), is_initialized(false) {}
};

#define PCK_HEAD_FOOT_SIZE 16UL
#define PCK_HEADER_MAGICNUM 0x98ac1123bb1543aaUL

struct PacketHeader
{
    uint64_t magic_num;
    uint64_t length;
};

#define BACKLOG_LEN 16 

static const timeval SELECT_TIMEOUT = { 0, 500000 };


//////////////////
// LOW LEVEL
//////////////////

static Result set_sockopt( const SocketData* sock, int level, int optname, int is_on )
{
    int opts = is_on;
    int irv = setsockopt( sock->fd, level, optname, &opts, sizeof opts );

    if ( irv != 0 ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_FATAL, "setsockopt() returned error '%s'", strerror( errno ) );
        return RV_UNSPECIFIED;
    } else {
        return RV_OK;
    }
}

/**
 * @param[out] sock The resulting server socket.
 */
static void initialize_server( SocketData* sock, uint16_t port, int backlog_len )
{
    int irv;
    assert( sock );
    sock->fd = 0;

    LOG_VAR_MSG( MDL_NETWORK, SVR_INIT, "Creating server socket on port %hu.", port );

    try { 

        sock->fd = socket( PF_INET, SOCK_STREAM, 0 );
        if ( sock->fd < 0 ) {
            LOG_VAR_MSG( MDL_NETWORK, SVR_FATAL, "socket() returned error '%s'", strerror( errno ) );
            throw ResourceUnavailableException();
        }

        // reuse socket address
        Result rv = set_sockopt( sock, SOL_SOCKET, SO_REUSEADDR, 1 );
        if ( rv_failed( rv ) ) { throw AedException( rv ); }

        // set to nonblocking
        int opts = fcntl( sock->fd, F_GETFL, 0 );
        fcntl( sock->fd, F_SETFL, opts | O_NONBLOCK );

        sock->addr.sin_family = AF_INET;
        sock->addr.sin_addr.s_addr = htonl( INADDR_ANY );
        sock->addr.sin_port = htons( port );

        irv = bind( sock->fd, (sockaddr*) &sock->addr, sizeof sock->addr );
        if ( irv != 0 ) {
            LOG_VAR_MSG( MDL_NETWORK, SVR_FATAL, "bind() returned error '%s'", strerror( errno ) );
            throw ResourceUnavailableException();
        }

        irv = listen( sock->fd, backlog_len );
        if ( irv != 0 ) {
            LOG_VAR_MSG( MDL_NETWORK, SVR_FATAL, "listen() returned error '%s'", strerror( errno ) );
            throw ResourceUnavailableException();
        }

    } catch ( ... ) {
        close( sock->fd );
        throw;
    }
}

/**
 * @param[out] sock The resulting connection socket.
 */
static Result connect_to( SocketData* sock, const char* host, uint16_t port )
{
    assert( sock );

    LOG_VAR_MSG( MDL_NETWORK, SVR_INIT, "Conecting to '%s' on port %hu.", host, port );

    int irv;
    Result rv;
    hostent* hp;

    sock->fd = socket( PF_INET, SOCK_STREAM, 0 );
    if ( sock->fd < 0 ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "socket() returned error '%s'", strerror( errno ) );
        return RV_RESOURCE_UNAVAILABLE;
    }

    hp = gethostbyname( host );
    if ( !hp ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "gethostbyname() returned error '%s'", hstrerror( h_errno ) );
        goto FAIL;
    }

    sock->addr.sin_family = AF_INET;
    memcpy( &sock->addr.sin_addr, hp->h_addr, hp->h_length);
    sock->addr.sin_port = htons( port );

    irv = connect( sock->fd, (sockaddr*) &sock->addr, sizeof sock->addr );
    if ( irv != 0 ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "connect() returned error '%s'", strerror( errno ) );
        goto FAIL;
    }

    rv = set_sockopt( sock, SOL_TCP, TCP_NODELAY, 1 );
    if ( rv_failed( rv ) ) { goto FAIL; }

    return RV_OK;

  FAIL:
    close( sock->fd );
    return RV_UNSPECIFIED;
}

static Result accept_connection( SocketData* sockrv, const SocketData* server )
{
    socklen_t addrlen = sizeof sockrv->addr;
    sockrv->fd = accept( server->fd, (sockaddr*) &sockrv->addr, &addrlen );

    if ( !sockrv->fd ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "accept() returned error '%s'", strerror( errno ) );
        return RV_UNSPECIFIED;
    }

    Result rv = set_sockopt( sockrv, SOL_TCP, TCP_NODELAY, 1 );
    if ( rv_failed( rv ) ) { return rv; }

    LOG_VAR_MSG( MDL_NETWORK, SVR_INIT, "Accepting new connection." );

    return RV_OK;
}

static Result close_socket( const SocketData* sock )
{
    assert( sock );
    int irv = close( sock->fd );
    if ( irv != 0 ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "socket close() returned error '%s'", strerror( errno ) );
        return RV_UNSPECIFIED;
    } else {
        LOG_VAR_MSG( MDL_NETWORK, SVR_INIT, "Connection %d terminated.", sock->fd );
        return RV_OK;
    }
}

static Result send_data( const SocketData* sock, void* datav, size_t len )
{
    uint8_t* data = (uint8_t*) datav;
    size_t total_sent = 0;

    while ( total_sent < len ) {
        ssize_t sent = send( sock->fd, data + total_sent, len - total_sent, MSG_NOSIGNAL );
        if ( sent <= 0 ) {
            return RV_UNSPECIFIED;
        }
        total_sent += sent;
    }

    return RV_OK;
}

static Result send_data_packet( const SocketData* sock, void* data, size_t len )
{
    assert( sock );

    Result rv;
    PacketHeader header;
    header.magic_num = PCK_HEADER_MAGICNUM;
    header.length = len;

    rv = set_sockopt( sock, SOL_TCP, TCP_CORK, 1 );
    if ( rv_failed( rv ) ) { return rv; }

    // send header with magic number of length of middle data
    rv = send_data( sock, &header, sizeof header );
    if ( rv_failed( rv ) ) {
        goto FAIL;
    }

    // send data
    rv = send_data( sock, data, len );
    if ( rv_failed( rv ) ) {
        goto FAIL;
    }

    rv = set_sockopt( sock, SOL_TCP, TCP_CORK, 0 );
    if ( rv_failed( rv ) ) { return rv; }

    return RV_OK;

  FAIL:
    LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "send() failed with error '%s'", strerror( errno ) );
    return RV_UNSPECIFIED;
}

static void build_fd_set( const SocketMap& sockets, fd_set* set, int* highest_fd )
{
    //LOG_VAR_MSG( MDL_NETWORK, SVR_TRIVIAL, "There are %zu active sockets", sockets.size() );

    FD_ZERO( set );
    int highest = 0;

    for ( SocketMap::const_iterator i = sockets.begin(); i != sockets.end(); ++i ) {
        int fd = i->second.fd;
        assert( fd >= 0 && fd < FD_SETSIZE );
        if ( fd != 0 ) {
            assert( !FD_ISSET( fd, set ) );
            FD_SET( fd, set );
            highest = std::max( highest, fd );
        }
    }

    *highest_fd = highest + 1;
}


///////////////////////
// SM STATE MANAGEMENT
///////////////////////


SocketManager::SocketManager()
{
    impl = new Impl();
    impl->is_initialized = false;
}

SocketManager::~SocketManager()
{
    assert( impl );

    // cleanup
    if ( impl->is_initialized ) {
        if ( impl->type == SM_SERVER ) {
            close_socket( &impl->my_sock );
        }
        for ( SocketMap::const_iterator i = impl->sockets.begin(); i != impl->sockets.end(); ++i ) {
            close_socket( &i->second );
            // don't need to bother cleaning up pool since it'll be deleted momentarily
        }

        assert( impl );
        delete impl->pool;
    }

    delete impl;
}

static void general_init( SocketManager::Impl* impl, size_t mpl, size_t mc )
{
    LOG_VAR_MSG( MDL_NETWORK, SVR_INIT, "Creating network manager with max conn = %zu and max packet len = %zu", mc, mpl );

    boost::mutex::scoped_lock lock( impl->state_mutex );
    if ( impl->is_initialized ) {
        throw InvalidOperationException();
    }

    impl->is_initialized = true;
    impl->is_running = false;

    impl->socket_counter = 1;

    impl->max_packet_len = mpl;
    impl->max_connections = mc;
    impl->buffer_size = mpl + PCK_HEAD_FOOT_SIZE;

    impl->pool = new common::util::FixedPool( impl->max_connections, impl->buffer_size );
}

void SocketManager::initialize( IServer* server, uint16_t port, size_t max_packet_len, size_t max_connections )
{
    // XXX doesn't properly clean up resources on initialization failure
    assert( server );

    try {

        general_init( impl, max_packet_len, max_connections );

        impl->my_sock.buffer = NULL;
        initialize_server( &impl->my_sock, port, BACKLOG_LEN );

        impl->type = SM_SERVER;
        impl->server = server;

    } catch ( ... ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_FATAL, "Failed to initialize server." );
        throw;
    }
}

void SocketManager::initialize( IClient* client, size_t max_packet_len, size_t max_connections )
{
    // XXX doesn't properly clean up resources on initialization failure
    assert( client );

    try {

        general_init( impl, max_packet_len, max_connections );

        impl->type = SM_CLIENT;
        impl->client = client;

    } catch ( ... ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_FATAL, "Failed to initialize server." );
        throw;
    }
}


/**
 * @post The entry in the map will have zeroed out fd and addr.
 *   Until these are filled, the select loop will not start listening to them.
 */
static Result add_new_neighbor( Socket** handle, SocketManager::Impl* impl )
{
    SocketData empty_socket;
    memset( &empty_socket, 0, sizeof empty_socket );

    LOG_MSG( MDL_NETWORK, SVR_NORMAL, "Adding new neighbor." );

    // lock structure
    boost::mutex::scoped_lock lock( impl->state_mutex );

    // make sure we aren't exceeding max connections
    if ( impl->sockets.size() >= impl->max_connections ) {
        LOG_MSG( MDL_NETWORK, SVR_ERROR, "requested network connection exceeds maximum allowed connections, ignoring." );
        return RV_INVALID_OPERATION;
    }

    // add to data structure, create buffer
    // make sure handle doesn't already exist (could happen with integer overflow)
    // this won't loop forever unless max connections is 2 ^ (sizeof size_t), which itself impossible
    do {
        *handle = (Socket*) (impl->socket_counter++);
    } while ( *handle == 0 || impl->sockets.find( *handle ) != impl->sockets.end() );

    try {
        bool was_added = impl->sockets.insert( std::make_pair( *handle, empty_socket ) ).second;
        assert( was_added );
    } catch ( ... ) {
        LOG_MSG( MDL_NETWORK, SVR_ERROR, "unable to add socket to map data structure." );
        return RV_OUT_OF_MEMORY;
    }

    // create buffer after adding to structure so we don't have to worry about
    // freeing memory on failure if addition to map fails
    uint8_t* buf = (uint8_t*) impl->pool->allocate();
    assert( buf );
    assert( impl->sockets.find( *handle ) != impl->sockets.end() );
    impl->sockets[*handle].buffer = buf;

    return RV_OK;
}

static void mark_neighbor_addr( SocketManager::Impl* impl, Socket* handle, SocketData* sock )
{
    boost::mutex::scoped_lock lock( impl->state_mutex );
    assert( impl->sockets.find( handle ) != impl->sockets.end() );
    SocketData& ref = impl->sockets[handle];
    ref.fd = sock->fd;
    ref.addr = sock->addr;
}

/**
 * @returns The file descriptor of the socket.
 */
static void remove_neighbor( SocketData* datap, SocketManager::Impl* impl, Socket* handle )
{
    // does NOT close socket, this is expected to be done elsewhere

    boost::mutex::scoped_lock lock( impl->state_mutex );

    SocketMap::iterator iter = impl->sockets.find( handle );
    assert( iter != impl->sockets.end() );

    SocketData data = iter->second;
    if ( datap ) {
        *datap = data;
    }

    size_t old_size = impl->sockets.size();
    impl->sockets.erase( iter );
    assert( old_size - 1 == impl->sockets.size() );
    impl->pool->free( data.buffer );
}

Result SocketManager::open_connection( Socket** sockrv, const char* host, uint16_t port )
{
    SocketData sock;
    Socket* handle;
    Result rv;

    // create neighbor
    rv = add_new_neighbor( &handle, impl );
    if ( rv_failed( rv ) ) {
        return rv;
    }

    // open connection
    rv = connect_to( &sock, host, port );
    if ( rv_failed( rv ) ) {
        remove_neighbor( NULL, impl, handle );
        return rv;
    }

    // set fd and addr in map
    mark_neighbor_addr( impl, handle, &sock );

    *sockrv = handle;
    return RV_OK;
}

void SocketManager::close_connection( Socket* sock )
{
    SocketData data;
    remove_neighbor( &data, impl, sock );
    close_socket( &data );
}

Result SocketManager::send_packet( Socket* sock, void* data, size_t len )
{
    // XXX NOT THREAD SAFE

    boost::mutex::scoped_lock lock( impl->state_mutex );
    SocketMap::iterator iter = impl->sockets.find( sock );
    if ( iter != impl->sockets.end() ) {
        SocketData& ref = iter->second;
        lock.unlock();
        // send packet
        return send_data_packet( &ref, data, len );
    } else {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "Socket %p not found, not sending message", sock );
        return RV_INVALID_ARGUMENT;
    }
}


//////////////
// LOOP
//////////////


static void process_incoming_connection( SocketManager::Impl* impl )
{
    SocketData incoming;
    Socket* handle;
    Result rv;

    // no way to "reject" connections with BSD sockets, so accept no matter what,
    // then just close immediately if too many connections
    rv = accept_connection( &incoming, &impl->my_sock );
    if ( rv_failed( rv ) ) {
        return;
    }

    rv = add_new_neighbor( &handle, impl );
    if ( rv_failed( rv ) ) {
        // just close immediately
        close_socket( &incoming );
        return;
    }

    mark_neighbor_addr( impl, handle, &incoming );

    rv = impl->server->accept_connection( handle );
    if ( rv_failed( rv ) ) {
        LOG_MSG( MDL_NETWORK, SVR_WARNING, "IServer returned error code when accepting new connection" );
        remove_neighbor( NULL, impl, handle );
        close_socket( &incoming );
    }
}

/// returns true if there is more processing to be done
static bool process_packet( SocketManager::Impl* impl, Socket* handle, SocketData* sock, bool* keep_alive )
{
    PacketHeader* header;
    Result rv;

    // XXX ignore byte ordering since both machines have same ordering

    // if no header is buffered, return false
    if ( sock->bytes_buffered < sizeof *header ) {
        return false;
    }

    // read header to determine packet length
    header = (PacketHeader*) sock->buffer;
    size_t total_length = header->length + (sizeof *header);

    // make sure the packet is small enough to fit in the buffer
    if ( total_length > impl->buffer_size ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "socket received too large of packet from %d, closing connection", sock->fd );
        *keep_alive = false;
        return false;
    }

    // if entire packet has not yet arrived, return
    if ( sock->bytes_buffered < total_length ) {
        return false;
    }

    // process packet
    rv = impl->get_network_object()->on_packet_received( handle, sock->buffer + sizeof *header, header->length );
    if ( rv_failed( rv ) ) {
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "INetworkObject returned error processing packet from %d, closing connection", sock->fd );
        *keep_alive = false;
        return false;
    }

    // shift remaining data down to front of buffer
    memmove( sock->buffer, sock->buffer + total_length, sock->bytes_buffered - total_length );
    sock->bytes_buffered -= total_length;

    return true;
}

/// Returns false if the socket should be closed.
static bool process_incoming_data( SocketManager::Impl* impl, SocketMap::iterator iter )
{
    SocketData& sock = iter->second;

    LOG_VAR_MSG( MDL_NETWORK, SVR_TRIVIAL, "Processing incoming data from socket %d", (int)sock.fd );

    // XXX this should NOT block, so if socket isn't nonblocking, need to pass correct flag
    ssize_t num_received = recv( sock.fd, sock.buffer + sock.bytes_buffered, impl->buffer_size - sock.bytes_buffered, 0 );
    if ( num_received < 0 ) { 
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "recv() on socket %d returned with error '%s'", sock.fd, strerror( errno ) );
        return false;
    } else if ( num_received == 0 ) {
        // EOF
        LOG_VAR_MSG( MDL_NETWORK, SVR_ERROR, "recv() on socket %d returned no bytes (EOF)", sock.fd );
        return false;
    }
    sock.bytes_buffered += num_received;

    // process arrived packets
    bool keep_alive = true;
    while ( process_packet( impl, iter->first, &sock, &keep_alive ) );
    
    LOG_VAR_MSG( MDL_NETWORK, SVR_TRIVIAL, "Done processing incoming data, size was %d", (int)num_received );
    return keep_alive;
}

/// Loops on select until socket manager is terminated.
static void select_loop( SocketManager* sockmgr, SocketManager::Impl* impl )
{
    if ( rv_failed( impl->get_network_object()->initialize( sockmgr ) ) ) {
        LOG_MSG( MDL_NETWORK, SVR_FATAL, "INewtorkObject returned error in initialize, aborting\n" );
        abort();
    }

    while ( impl->check_is_running() ) {

        LOG_VAR_MSG( MDL_NETWORK, SVR_TRIVIAL, "Top of select loop" );

        // fill timeval, read sets
        timeval timeout = SELECT_TIMEOUT;
        fd_set read_set;
        int highest_fd;
        boost::mutex::scoped_lock lock( impl->state_mutex );
        // must build inside lock to prevent connection from being added during build
        build_fd_set( impl->sockets, &read_set, &highest_fd );
        // if this is a server, also add self
        if ( impl->type == SM_SERVER ) {
            FD_SET( impl->my_sock.fd, &read_set );
            highest_fd = std::max( highest_fd, impl->my_sock.fd + 1 );
        }
        lock.unlock();

        LOG_VAR_MSG( MDL_NETWORK, SVR_TRIVIAL, "Invoking select()" );

        // block until timeout or activity
        int num_ready = select( highest_fd, &read_set, NULL, NULL, &timeout );

        LOG_VAR_MSG( MDL_NETWORK, SVR_TRIVIAL, "select() retruned." );

        if ( num_ready < 0 ) {
            LOG_VAR_MSG( MDL_NETWORK, SVR_FATAL, "select() returned error '%s'", strerror( errno ) );
            abort();
        }

        if ( impl->type == SM_SERVER && FD_ISSET( impl->my_sock.fd, &read_set ) ) {
            // accept incoming connection
            process_incoming_connection( impl );
        }

        for ( SocketMap::iterator i = impl->sockets.begin(); i != impl->sockets.end(); ++i ) {
            Socket* handle = i->first;
            int fd = i->second.fd;
            if ( FD_ISSET( fd, &read_set ) ) {
                // accept incoming data
                bool do_kill = !process_incoming_data( impl, i );
                if ( do_kill ) {
                    SocketData data;
                    remove_neighbor( &data, impl, handle );
                    LOG_VAR_MSG( MDL_NETWORK, SVR_NORMAL, "killing marked socket '%d'", data.fd );
                    close_socket( &data );
                    // XXX doesn't check return value
                    impl->get_network_object()->on_connection_closed( handle );

                }

                // only process one thing per loop since the process data call could have
                // messed with internal data structures
                break;
            }
        }
    }
}

boost::thread* SocketManager::run()
{ 
    boost::mutex::scoped_lock lock( impl->state_mutex );

    if ( !impl->is_initialized || impl->is_running ) {
        return NULL;
    }

    impl->is_running = true;
    lock.unlock();

    return new boost::thread( boost::bind( select_loop, this, impl ) );
}

void SocketManager::terminate()
{ 
    boost::mutex::scoped_lock lock( impl->state_mutex );
    impl->is_running = false;
}

} // network
} // common
} // aed

