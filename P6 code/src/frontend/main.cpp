#include <iostream>
#include <getopt.h>
#include "frontplane.hpp"
#include "common/log.hpp"

using namespace aed;
using namespace aed::frontend;

int main( int argc, char **argv )
{
	Frontplane frontplane;
    const char* hostname;
    uint16_t port;

    if ( argc != 3 ) {
        goto FAIL;
    }

    hostname = argv[1];
    if (1 != sscanf( argv[2], "%hu", &port ) ) {
        goto FAIL;
    }

    // Initialize the logger
    log_init( "felog.txt", stdout );

    if ( !frontplane.initialize( "config/fe_config.xml", "config/shared_config.xml", 0, "") ) {
        LOG_MSG(MDL_FRONT_END, SVR_ERROR, "Frontplane initialization failed." );
        return 1;
    }


    frontplane.start( hostname, port );

    frontplane.destroy();

    return 0;

  FAIL:

    printf("Usage: %s <backend hostname or ip> <backend port>\n");
    return 1;

}

