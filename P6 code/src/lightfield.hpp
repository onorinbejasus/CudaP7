
#ifndef LF_LIGHTFIELD_HPP
#define LF_LIGHTFIELD_HPP

#include <stdint.h>
#include <cuda_runtime.h>

namespace lightfield {

struct LFCamera
{
    float3 position;
    float3 direction;
    float3 up;
    // vertical field of view, in radians.
    float fov;
    // width / height
    float aspect_ratio;
};

struct LFData
{
    typedef uint16_t LFindex[3];
    typedef uint8_t LFabg[3];

    // The radius of the lightfield. All cameras are this distance
    // from the lightfield center.
    float camera_radius;
    // The vertical field of view of each camera, in radians.
    float camera_fov;
    // The aspect ratio of each camera, (width/height).
    float camera_aspect;

    // The dimensions of the indices and abg arrays.
    size_t lookup_width;
    size_t lookup_height;

    // The total number of cameras used in the lightfield.
    size_t num_cameras;
    // the dimension of each camera's texture.
    size_t camera_tex_width;
    size_t camera_tex_height;

    // A 3D array of all the camera data.
    // Is the concatenation of each row-major camera texture.
    //
    // The index of the pixel of camera k at texture coordinates (s,t) is
    // "k * camera_tex_width * camera_tex_height + s + t * camera_tex_width".
    //
    // The value is a 4-byte RGBA.
    uint32_t* color_textures;

    // The nearest 3 cameras to a given point on the lightfield sphere,
    // as a 2D texture in spherical coordinates.
    //
    // The array is row-major with width lookup_width and height lookup_height,
    // so the entry at 2D index (s,t) is "indices[s + t * lookup_width]".
    //
    // Array lookup:
    // Given the longitude (theta) and latitude (phi) of a point on the sphere,
    // the corresponding texture coordinate into the array is 
    // (theta/(2*PI) * lookup_width - 0.5, phi/PI * 2 * lookup_height - 0.5).
    //
    // Note you need "2 * lookup_height" rather than "lookup_height" since
    // the lookup texture only spans the top hemisphere.
    //
    // The values in LFindex are indices into the "camera_positions" array.
    LFindex* indices;

    // The barycentric coordinates for the cameras described in the
    // corresponding entry of "indices".
    // Array indexing and size are indentical to "indices".
    // Values are floats stored as bytes, i.e., the float value for
    // each entry "x" is "x/255.0f";
    LFabg* abg;

    // Position of each camera as a 1D array of "num_cameras" length.
    // The position is relative to the center of the lightfield sphere,
    // and it has unit length.
    //
    // So, for the camera at index k:
    // - the position is "camera_positions[k] * radius"
    // - the direction is "-camera_positions[k]"
    // - the up vector is NOT given in the data. A function that computes the
    //   up vector is given in lightfield.cu.
    float3* camera_positions;
};

// The interface you must implement for your assignment
class ILightfield
{
public:
    virtual ~ILightfield() { }

    // Initializes your renderer.
    // data: The lightfield data, see the struct for more detail.
    // window_width: The window width, which is the maximum dimension of each query.
    // window_height: The window height, which is the maximum dimension of each query.
    virtual bool initialize( const LFData* data, size_t window_width, size_t window_height ) = 0;

    // Render the lightfield with the given camera, storing the result in the given buffer.
    // color: The CPU buffer in which to output the color data.
    // width: The width of the buffer.
    // height: The height of the buffer.
    // camera: The viewing camera to use to query the lightfield.
    virtual void render( uint8_t* color, size_t width, size_t height, const LFCamera* camera ) = 0;
};

// Creates an implementation of ILightField, allocated with "new".
// TOOD to be implemented by you
ILightfield* create_lightfield();

}

#endif

