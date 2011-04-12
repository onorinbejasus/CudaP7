/** 
 * @file    backplane.hpp
 * @brief   This class is the backplane of the whole frontend program. It contains the sharing queues 
 *			, the working modules and the thread pool, the class also initialize the working environment
 *			of SDL and openGL
 *			
 * @author  Wei-Feng Huang
 * @date    July 8, 2010
 */
#ifndef AED_FRONTPLANE_HPP
#define AED_FRONTPLANE_HPP

#include "frontend/stage/analyzer.hpp"
#include "common/network/socket.hpp"

namespace aed {
namespace frontend {

enum FrontplaneMode
{
    BPM_NORMAL,
    BPM_CAPTURE,
    BPM_MOVIE,
    BPM_PROCESS
};

class Frontplane
{
public:
    Frontplane()  {}
    ~Frontplane() {}

    // Initialize application
    bool initialize(char* fe_config_file, char* be_config_file, int fe_index, char* comment);

    void activate_special_mode( FrontplaneMode mode, const char* filename, size_t data );

    // Start the application. There is a loop inside this function. 
    // This function cannot finish until the application is closed. 
    void start( const char* hostname, uint16_t port );

    // Destroy application
    void destroy();

private:
    Frontplane(const Frontplane&);
    void operator=(const Frontplane&);

    stage::Analyzer::GeometryDataList geometries;
    common::network::SocketManager socket_manager;

    bool parse_config_files(char* fe_config_file, char* shared_config_file);

    FrontplaneMode mode;
    const char* mode_file;
    size_t mode_data;
};

} /* frontend */
} /* aed */

#endif
