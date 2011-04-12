#ifndef AED_PIPELINE_QUEUE_HPP
#define AED_PIPELINE_QUEUE_HPP

#include "common/result.hpp"
#include "common/log.hpp"

#include <stdint.h>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <typeinfo>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

namespace aed {
namespace common {
namespace pipeline {


class CircularQueue
{
public:
    CircularQueue();
    ~CircularQueue();

    Result initialize( size_t capacity, size_t elem_size );

    // The following functions are thread-safe

    size_t capacity() const;
    size_t elem_size() const;
    size_t queue_size() const;
    size_t alloc_size() const;

    /// Pops the top element off the queue, returning the pointer to that element.
    /// Will block until there is an item in the queue or abort_wait is called.
    Result block_until_pop( void** elemp );

    /// Allocates space for a new object, returning the pointer to that element.
    /// Will block until there is available space in the queue or abort_wait is called.
    Result block_until_allocate( void** elemp );

    /// Releases a popped or allocated object, does not block.
    /// Can be used to both release popped objects once processing is complete
    /// or to cancel a pending push of an allocated object.
    Result release( void* elem );

    /// Pushes an allocated to the queue, does not block.
    /// Object must not already be in the queue.
    Result push( void* elem );

    /// Releases all blocking threads with the return value RV_OPERATION_ABORTED.
    /// All subsequent function calls will return RV_INVALID_OPERATION.
    void abort_wait();

    // end thread-safe block

    struct Data;

private:

    Data* data;
};

// HACK TODO we really only want this class to work with PODS, but Vector2, Camera, etc are not pods, so the
// render request doesn't work. In the meanwhile, hopefully this constructor/destructor code is actually correct...
// empty shell class so CircularQueue can masquerade as an opaque object that can only be pushed to or aborted.
template<typename T>
class PipelineQueue
{
public:

    PipelineQueue() : is_initialized( false ) { }

    ~PipelineQueue() { }

    void initialize( size_t capacity )
    {
        Result rv = cq.initialize( capacity, sizeof( T ) );
        if ( rv_failed( rv ) ) {
            throw AedException( rv );
        }
        is_initialized = true;
    }

    void abort_wait() { cq.abort_wait(); }

    CircularQueue* get_queue() { return is_initialized ? &cq : NULL; }

private:
    PipelineQueue(const PipelineQueue&);
    void operator=(const PipelineQueue&);

    CircularQueue cq;
    bool is_initialized;
};


template<typename T>
class QueuePusher
{
public:
    explicit QueuePusher(PipelineQueue<T>& pq)
      : cq( pq.get_queue() ),
        elem(0),
        wasPushed(false)
    {
        if ( !this->cq ) {
            LOG_VAR_MSG( MDL_PIPELINE, SVR_ERROR, "Queue is not initialized" );
            throw InvalidOperationException("queue not initialized");
        }

        void* p;
        // try to allocate, block until succeeds
        Result rv = this->cq->block_until_allocate( &p );
        LOG_VAR_MSG( MDL_PIPELINE, SVR_DEBUG3, "allocated space for '%s' at %p", typeid( T ).name(), p );

        if ( rv == RV_OPERATION_ABORTED || rv == RV_INVALID_OPERATION ) {
            throw InvalidOperationException("queue aborted");
        }

        assert(!rv_failed(rv)); /*&& "queue allocation failed" */

        //LOG_VAR_MSG( MDL_PIPELINE, SVR_TRIVIAL, "invoking ctor for '%s'", typeid( T ).name() );

        this->elem = new (p) T();

        //LOG_VAR_MSG( MDL_PIPELINE, SVR_TRIVIAL, "finished allocation for '%s'", typeid( T ).name() );
    }

    void push()
    {
        Result rv = this->cq->push( elem );
        LOG_VAR_MSG( MDL_PIPELINE, SVR_DEBUG3, "'%s' has pushed item %p", typeid( T ).name(), elem );
        if ( rv != RV_INVALID_OPERATION ) {
            assert(!rv_failed(rv)); /* && "queue complete invalid" */
        }

        wasPushed = true;
    }
    
    ~QueuePusher()
    {
        // if not pushed, cancel operation
        if ( !wasPushed ) {
            elem->~T();
            Result rv = this->cq->release( elem );
            if ( rv != RV_INVALID_OPERATION ) {
                assert(!rv_failed(rv)); /* && "queue cancel invalid" */
            }
        }
    }

    T* operator->() { return elem; }
    T& operator*() { return *elem; }

private:
    QueuePusher(const QueuePusher&);
    void operator=(const QueuePusher&);

    CircularQueue* cq;
    T* elem;
    bool wasPushed;
};



} // pipeline
} // common
} // aed

#endif /* AED_PIPELINE_QUEUE_HPP */

