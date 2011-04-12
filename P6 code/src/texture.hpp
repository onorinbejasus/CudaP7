
#ifndef LF_TEXTURE_HPP
#define LF_TEXTURE_HPP

#include "lightfield.hpp"

namespace lightfield {

enum LFTextureSize { LFTS_128, LFTS_256 };
enum LFNumCameras { LFNC_545, LFNC_2113 };

// returns true on success, false on failure
bool load_lightfield( LFData* result, const char* root_data_dir, LFNumCameras num_cameras, LFTextureSize texture_size );

}

#endif

