/** @file config.hpp
 * 
 *  This file defines the coniguration of some information
 *  for the whole project: the compression format, the picture width/height
 *  the size of the buffer...etc.
 *
 */

#ifndef AED_CONFIG_HPP
#define AED_CONFIG_HPP
 
#include "common/log.hpp"
#include "common/render_types.hpp"
#include "common/network/packet.hpp"
#include "common/util/timer.hpp"
#include "common/math/camera.hpp"
#include "common/math/geometry.hpp"
#include <vector>
#include <stdint.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

namespace aed {
namespace frontend {

// The maximum number of concurrent frames
#define MAX_FRAMES_IN_FLIGHT 20
#define MAX_OBJECTS 128

#define FLUID_DIM 3

/**
 * Configuration records all configuration information for frontend program.
 * Configuration is a singleton class.
 *
 * NOTE: DO NOT MOVE THIS CLASS DOWN!
 * This class should be defined at the top of this file, since many other
 * structures might rely on frontend configuration data, like window size.
 */
class Configuration
{
public:
    Configuration() {};
    ~Configuration() {};

    struct Backend {
        std::string hostname;
        uint16_t port;
    };

    int frontend_index;

    std::vector<Backend> backends;

    int num_geometry;               // the number of objects in the scene

    size_t window_width;
    size_t window_height; 

    size_t frames_in_flight;
    size_t target_framerate;
    size_t num_decompressors;
    
    size_t rendering_fluid;
    size_t fluid_dimension;
    size_t enable_user_input;

    uint32_t window_pixel_number() const { return window_width * window_height;  }
    uint32_t window_RGBA_size() const    { return window_pixel_number() * 4;     }
    uint32_t max_image_size() const      { return window_RGBA_size();            }
 
    std::string maple_shadow;
    std::string chestnut_shadow;
    std::string oak_shadow;
    std::string plum_shadow;
    std::string maple_dark_shadow;

    std::string env_grid_model;
    std::string env_shadow_plane_model;
    std::string env_grid_texture;

    std::string fluid_smoke_texture;

    bool phonyMode;
    
    // For timing information
    bool gather_timing_data;
    common::util::TimerTypeHandle feMemoryHandle;
    common::util::TimerTypeHandle compositorTimerHandle;
    common::util::TimerTypeHandle compositorObjRecTimerHandle;
    common::util::TimerTypeHandle receiverTimerHandle;
    common::util::TimerTypeHandle receiverTimerHandle2;
    common::util::TimerTypeHandle analyzerTimerHandle;
    common::util::TimerTypeHandle analyzerObjectTimerHandle;
    common::util::TimerTypeHandle decompressorTimerHandle;
};

extern const Configuration& feconfig;

struct ObjectRenderData
{
    // The pass in which to render this object, zero indicates N/A
    size_t pass_num;
    // Whether or not this object has a shadow and its type
    // zero indicates none or N/A
    size_t shadow_type;
    // the rendering algorithm
	common::RenderAlgType type;
};


struct MutexCvarPair
{
    boost::mutex mutex;
    boost::condition cvar;
    bool is_aborted;
};

/// Object information, shared by the analyzer and compositor.
struct ObjectInfo
{
    //// to be set by analyzer ////

    // assumes rendering order and backend id matches index in list, so these are not entries

    // the frontend's object id
    uint32_t object;
    // the clipping data
    ClipRect clip;
    // the object's transformation
    Transform transform;
    // rendering data
    ObjectRenderData rdata;
};

typedef std::vector<ObjectInfo> ObjectList;

/// Frame information, shared by the analyzer and compositor.
struct FrameInfo
{
    //// to be set by analyzer ////

    // the frame id
    size_t frame;
    // the objects, sorted back to front
    ObjectList objects;
    // the camera, in world space
    Camera camera;

    //// to be set by compositor ////

    // set to true when the frame is finished rendering. a new frame
    // should not be started until this is set to true.
    bool is_rendered;
};

/* --- define maximum window size --- */
#define MAX_WINDOW_WIDTH 1024
#define MAX_WINDOW_HEIGHT 1024
#define MAX_BUFFER_SIZE ((1 << 21) - sizeof(BackendResponseHeader))

struct ResultData
{
    BackendResponseHeader header;
    uint8_t data[MAX_BUFFER_SIZE];
    // we need this because otherwise, the default ctor takes 10 ms to run (???)
    ResultData() { }
};

struct CompositorInput
{
    enum { CI_FrameInfo, CI_ResultData } type;
    FrameInfo* frame_info;
    ResultData result_data;
};

typedef size_t BACKEND_ID;

struct ProcessingImage {
    uint32_t clip_width;
    uint32_t clip_height;
    uint32_t x_offset;
    uint32_t y_offset;
	uint32_t image_data_size;
	uint8_t *image_data;
};
typedef std::vector< ProcessingImage > ProcessingImageList;




} /* frontend */
} /* aed */

#endif /* AED_CONFIG_HPP */

