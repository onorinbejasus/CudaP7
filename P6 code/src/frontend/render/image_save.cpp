
#include "image_save.hpp"
#include "common/util/imageio.hpp"
#include "frontend/opengl.hpp"

namespace aed {
namespace frontend {
namespace render {

// Wraps the general functionality of saving an image and writes the current
// frame buffer to a specified file name.  Also returns true on succces,
// false otherwise.
bool imageio_save_screenshot( const char *filename, int width, int height )
{
    unsigned char *buffer = new unsigned char[width * height * 4];
    if (!buffer)
        return false;
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
    bool result = common::util::imageio_save_image(filename, buffer, width, height);
    delete [] buffer;
    return result;
}

} // render
} // frontend
} // aed

