
#include "result.hpp"
#include "common/log.hpp"

namespace aed {

static const char* rv_string[_RV_MAX_] = {
    "Ok",
    "Unspecified",
    "Invalid Argument",
    "Out of Memory",
    "Overflow",
    "Resource Unavailable",
    "Parse Error",
    "Invalid Operation",
    "Operation Aborted",
    "Unimplemented",
};

const char* rv_error_string( Result rv )
{
    assert( rv >= 0 && rv < _RV_MAX_ );
    return rv_string[rv];
}

AedException::AedException( Result rv, const char* msg ) throw()
  : rv(rv), msg(msg)
{
    LOG_VAR_MSG( MDL_CORE, SVR_WARNING, "Exception of type '%s' created, msg='%s'", rv_error_string( rv ), this->what() );
}
    
const char* AedException::what() const throw() 
{
    return msg == NULL ? rv_error_string( rv ) : msg;
}

} /* aed */

