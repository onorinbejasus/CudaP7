
#include "common/log.hpp"

#include "common/pipeline/queue.hpp"
#include "common/util/fixed_pool.hpp"

#include <boost/circular_buffer.hpp>


namespace aed {
namespace common {
namespace pipeline {

struct CircularQueue::Data
{
    Data(size_t capacity, size_t elem_size)
      : queue( capacity ),
        pool(capacity, elem_size),
        is_aborted( false )
    { }

    // queue for pipeline metadata
    boost::circular_buffer<void*> queue;
    // memory pool for pipeline objects
    common::util::FixedPool pool;

    // if true, the queue is now disabled
    bool is_aborted;

    boost::mutex mutex;
    boost::condition cvar_push;
    boost::condition cvar_pop;
};


CircularQueue::CircularQueue() : data( NULL ) { }

CircularQueue::~CircularQueue() { }

size_t CircularQueue::capacity() const
{
    return data->pool.capacity();
}

size_t CircularQueue::elem_size() const
{
    return data->pool.elem_size();
}

size_t CircularQueue::queue_size() const
{
    return data->queue.size();
}

size_t CircularQueue::alloc_size() const
{
    return data->pool.size();
}

Result CircularQueue::initialize( size_t capacity, size_t elem_size )
{
    assert( this->data == NULL );
    try {
        this->data = new Data( capacity, elem_size );
    } catch ( std::bad_alloc ) {
        return RV_OUT_OF_MEMORY;
    }

    return RV_OK;
}

Result CircularQueue::block_until_pop( void** elemp )
{
    assert( elemp );

    boost::mutex::scoped_lock lock( data->mutex );

    if ( data->is_aborted ) {
        return RV_INVALID_OPERATION;
    }

    // wait until data is in queue
    while ( data->queue.empty() ) {
        data->cvar_pop.wait( lock );
        // abort if necessary
        if ( data->is_aborted ) {
            return RV_OPERATION_ABORTED;
        }
    }

    void* ptr = data->queue.front();
    data->queue.pop_front();

    *elemp = ptr;
    return RV_OK;
}

Result CircularQueue::release( void* elem )
{
    assert( data->pool.is_in_pool( elem ) );

    boost::mutex::scoped_lock lock( data->mutex );

    if ( data->is_aborted ) {
        return RV_INVALID_OPERATION;
    }

    data->pool.free( elem );
    // now there is memory to allocate for pushing
    data->cvar_push.notify_one();

    return RV_OK;
}

Result CircularQueue::block_until_allocate( void** elemp )
{
    assert( elemp );

    boost::mutex::scoped_lock lock( data->mutex );

    if ( data->is_aborted ) {
        return RV_INVALID_OPERATION;
    }

    // wait until memory available
    while ( data->pool.size() == data->pool.capacity() ) {
        LOG_MSG( MDL_PIPELINE, SVR_TRIVIAL, "[WARN] out of space in queue, blocking" );
        data->cvar_push.wait( lock );
        // abort if necessary
        if ( data->is_aborted ) {
            return RV_OPERATION_ABORTED;
        }
    }

    void* ptr = data->pool.allocate();
    assert( ptr );

    *elemp = ptr;
    return RV_OK;
}

Result CircularQueue::push( void* elem )
{
    assert( data->pool.is_in_pool( elem ) );

    boost::mutex::scoped_lock lock( data->mutex );

    if ( data->is_aborted ) {
        return RV_INVALID_OPERATION;
    }

    assert( data->queue.size() < data->queue.capacity() );
    data->queue.push_back( elem );
    // now there is something to pop
    data->cvar_pop.notify_one();

    return RV_OK;
}

void CircularQueue::abort_wait()
{ 
    boost::mutex::scoped_lock lock( data->mutex );

    data->is_aborted = true;

    // wake up everyone
    data->cvar_pop.notify_all();
    data->cvar_push.notify_all();
}

} // pipeline
} // common
} // aed

