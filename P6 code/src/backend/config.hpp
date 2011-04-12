#ifndef AED_BACK_END_CONFIGURATION_HPP
#define AED_BACK_END_CONFIGURATION_HPP

#include "common/util/timer.hpp"
#include "common/render_types.hpp"
#include <stdint.h>
#include <string>

namespace aed {
namespace backend {

// GLOBAL DEFINITION OF RENDERING RESULT BUFFER SIZE
#define MAX_IMAGE_SIZE (1024*1024*4)
#define MAX_BUFFER_SIZE (MAX_IMAGE_SIZE*4)

//#define CAMERA_FOV 37.0

enum GPUOutputMode {
    GPU_OUTPUT_RGBA   = 0,
    GPU_OUTPUT_QUANTIZATION
};

enum GPUInputMode {
    GPU_INPUT_RGBA = 0,
    GPU_INPUT_DXT5,
    GPU_INPUT_QUANTIZATION
};

enum RenderMode{
    SHADER_MODE = 0,
    C_SIMULATION_MODE,  // 1
    NUM_MODE,           // 2
    TEST_LF_MODE        // 3, this is just for test
};

enum ShaderStage {
    RENDER_MODEL = 0,
    RENDER_THETA_PHI,           // 1
    RENDER_TEXTURE_INDEX,       // 2
    RENDER_ALPHA_BETA_GAMMA,    // 3
    RENDER_TEXTURE_ST_COORD,    // 4
    RENDER_TEXTURE_INDEX_MOD_4, // 5
    RENDER_1ST_TEXTURE,         // 6
    RENDER_2ND_TEXTURE,         // 7
    RENDER_3RD_TEXTURE,         // 8
    RENDER_BLENDING1,           // 9
    RENDER_BLENDING2,           // 10
    NUM_SHADER_STAGE            // 11
};


// List of methods for rendering depth information.
// Different methods require different distance equations.
enum DepthType
{
    ENVIRONMENT_FOG = 0,
    LUMINANCE_DEPTH = 1
};

struct LightfieldSettings {    
    int num_textures;
    char color_texture_path_format[1024];
    char depth_texture_path_format[1024];
    char index_lookup_filename[1024];
    char abg_lookup_filename[1024];
    char pos_lookup_filename[1024];
    DepthType depth_type;
    double lf_radius;
    double depth_near;
    double depth_dist;
};

} // namespace backend
} // namespace aed

#endif // AED_BACK_END_CONFIGURATION_HPP

