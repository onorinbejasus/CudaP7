#include <string.h>
#include "common/log.hpp"
#include "common/render_types.hpp"

namespace aed {
namespace common {

RenderAlgType get_render_type(const char* type_string)
{
    if(strncmp(type_string, "lightfield", 11) == 0) {
        return LIGHTFIELD;
    }
    else if(strncmp(type_string, "p2_fluid_sim", 13) == 0) {
        return P2_FLUID_SIM;
    }
    else {
        LOG_VAR_MSG( MDL_BACK_END, SVR_ERROR, "Don't know how to load render algorithm type \"%s\".\n", type_string);
        throw InvalidArgumentException("Error loading render algorithm type");
    }
}

}
}

