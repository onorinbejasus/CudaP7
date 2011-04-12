#ifndef AED_RENDER_TYPES_HPP
#define AED_RENDER_TYPES_HPP

namespace aed {
namespace common {

enum RenderAlgType {
    LIGHTFIELD   = 0,
    P2_FLUID_SIM = 1
};

RenderAlgType get_render_type(const char* type_string);

}
}

#endif // AED_RENDER_TYPES_HPP

