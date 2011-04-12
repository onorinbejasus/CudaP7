
#include "common/util/fixed_pool.hpp"

#include <cassert>
#include <stdint.h>
#include <algorithm>

namespace aed {
namespace common {
namespace util {

static inline void* elem_at(uint8_t* pool, size_t i, size_t capacity, size_t elem_size)
{
    assert(i < capacity);
    return pool + i * elem_size;
}

FixedPool::FixedPool( size_t capacity, size_t elem_size )
{
    assert(elem_size > 0 && capacity > 0);

    _orig_elem_size = elem_size;
    elem_size = std::max( elem_size, sizeof(size_t) );

    // guarantee 8-bit alignment
    this->_pool = (uint8_t*) new uint64_t[capacity * elem_size / 8 + 1];
    this->_capacity = capacity;
    this->_size = 0;
    this->_elem_size = elem_size;
    this->_free_list_head = 0;

    // generate free list, store next element
    for ( size_t i = 0; i < capacity; ++i ) {
        *((size_t*) elem_at(_pool, i, capacity, elem_size)) = i + 1;
    }
}

FixedPool::~FixedPool()
{
    delete [] _pool;
}

size_t FixedPool::capacity() const
{
    return _capacity;
}

size_t FixedPool::elem_size() const
{
    return _orig_elem_size;
}

size_t FixedPool::size() const
{
    return _size;
}

void* FixedPool::allocate()
{
    if ( _size == _capacity ) {
        throw std::bad_alloc();
    }

    void* next = elem_at(_pool, _free_list_head, _capacity, _elem_size);
    _free_list_head = *((size_t*) next);
    _size++;

    return next;
}

void FixedPool::free(void* vptr)
{
    uint8_t* ptr = (uint8_t*) vptr;

    assert( _size > 0 );
    assert( is_in_pool( vptr ) );

    size_t index = ( ptr - _pool ) / _elem_size;
    *((size_t*) ptr) = _free_list_head;
    _free_list_head = index;

    _size--;
}

bool FixedPool::is_in_pool( void* vptr )
{
    size_t ptr = (size_t) vptr;
    size_t base = (size_t) _pool;

    return ptr >= base
        && ptr < base + _capacity * _elem_size
        && (ptr - base) % _elem_size == 0;
}

} // util
} // common
} // aed

