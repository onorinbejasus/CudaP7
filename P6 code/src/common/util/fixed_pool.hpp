#ifndef AED_PIPELINE_FIXEDPOOL_HPP
#define AED_PIPELINE_FIXEDPOOL_HPP

#include <cstddef>
#include <stdint.h>

namespace aed {
namespace common {
namespace util {

// XXX use nothrow and return error on initialize

class FixedPool
{
public:
    FixedPool( size_t capacity, size_t elem_size );
    ~FixedPool();

    size_t capacity() const;
    size_t elem_size() const;
    size_t size() const;

    void* allocate();
    void free(void* ptr);

    bool is_in_pool(void* ptr);

private:
    FixedPool(const FixedPool&);
    void operator=(const FixedPool&);

    uint8_t* _pool;
    size_t _capacity;
    size_t _size;
    size_t _orig_elem_size; // the elem size before increasing to at least size_t
    size_t _elem_size;
    size_t _free_list_head;
};

} // util
} // common
} // aed

#endif /* AED_PIPELINE_FIXEDPOOL_HPP */

