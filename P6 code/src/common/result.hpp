/** @file result.hpp
 *
 *  An enumeration of error codes and a few useful functions for them.
 */

#ifndef _AED_RESULT_HPP_
#define _AED_RESULT_HPP_

#include <assert.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <exception>

namespace aed {

/// error codes
enum Result {
    RV_OK,                      // success
    RV_UNSPECIFIED,             // when no other error is appropriate
    RV_INVALID_ARGUMENT,        // generic one or more arguments were invalid
    RV_OUT_OF_MEMORY,           // insufficient memory/unable to allocate more // XXX we should use std::bad_alloc for exceptions
    RV_OVERFLOW,                // the operation would cause overflow
    RV_RESOURCE_UNAVAILABLE,    // file/socket/etc could not be opened
    RV_PARSE_ERROR,             // string/data failed to lex/parse/compile
    RV_INVALID_OPERATION,       // operation not permitted in object's current state
    RV_UNIMPLEMENTED,           // the function has not yet been implemented
    RV_OPERATION_ABORTED,       // the operation has been user-aborted
    RV_FATAL,                   // fatal error, if seen process should exit // XXX is this used?
    _RV_MAX_                    // unused, marks max value for array
};

/**
 * Returns the string version of each error, e.g., RV_OK returns "RV_OK".
 * @param rv The return value of which to get a string.
 * @return A string of the error. Will always be non-null. The string
 *  is statically allocated, do not free it.
 */
const char* rv_error_string( Result rv );

/**
 * @return True if the given code is an error, false if it is a sccess.
 */
inline bool rv_failed( Result rv )
{
    return rv != RV_OK;
}

class AedException : public std::exception
{
public:
    AedException( Result rv, const char* msg = NULL ) throw();
    virtual ~AedException() throw() { }
    virtual const char* what() const throw(); 
private:
    Result rv;
    const char* msg;
};

#define DEFINE_GENERIC_AED_EXCEPTION( name, rv ) \
    class name ## Exception : public AedException \
    { \
    public: \
        name ## Exception() throw() : AedException( rv ) { } \
        name ## Exception(const char* msg) throw() : AedException( rv, msg ) { } \
        virtual ~name ## Exception() throw() { } \
    }

DEFINE_GENERIC_AED_EXCEPTION( Unspecified, RV_UNSPECIFIED );
DEFINE_GENERIC_AED_EXCEPTION( InvalidArgument, RV_INVALID_ARGUMENT );
DEFINE_GENERIC_AED_EXCEPTION( ResourceUnavailable, RV_RESOURCE_UNAVAILABLE );
DEFINE_GENERIC_AED_EXCEPTION( ParseError, RV_PARSE_ERROR );
DEFINE_GENERIC_AED_EXCEPTION( InvalidOperation, RV_INVALID_OPERATION );
DEFINE_GENERIC_AED_EXCEPTION( Unimplemented, RV_UNIMPLEMENTED );
DEFINE_GENERIC_AED_EXCEPTION( OperationAborted, RV_OPERATION_ABORTED );

#undef DEFINE_GENERIC_AED_EXCEPTION

} /* aed */

#endif /* _AED_RESULT_HPP_ */

