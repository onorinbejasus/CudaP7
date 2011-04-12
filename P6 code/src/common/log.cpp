/** @file log.cpp
 *
 *  A module for logging/debug printing.
 */
#include <pthread.h>
#include "common/log.hpp"

#include <stdarg.h>
#include <time.h>
#include <sys/time.h>
#include <boost/thread/mutex.hpp>

namespace aed {


#define MAX_OUTPUTS 2

#define AED_SYNCHRONIZE_LOG_WRITES 1

#if AED_SYNCHRONIZE_LOG_WRITES
static boost::mutex mutex;
#endif

static const char* mdl_string[_MDL_MAX_] = {
    "Core",
    "Math",
    "Render",
    "Texture",
    "Pipeline",
    "Backend",
	"Frontend",
    "Network",
    "Compression"
};

static const char* svr_string[_SVR_MAX_] = {
    "FATAL",
    "ERROR",
    "WARN",
    "INFO",
    "DBG1",
    "DBG2",
    "DBG3"
};

static struct {
    Severity max_severity[_MDL_MAX_];
    FILE* outputs[MAX_OUTPUTS];
} log;

Result log_init( const char* logfile, FILE* print_stream )
{
    Result rv;

    static bool initialized = false;
    if ( initialized ) {
        rv = RV_INVALID_OPERATION;
        goto FAIL;
    }
    initialized = true;

    log_set_all_severity( SVR_INFO );

    log.outputs[0] = NULL;
    log.outputs[1] = print_stream;

    if ( logfile ) {
        // try to open a new logfile, print some basic data
        log.outputs[0] = fopen( logfile, "w" );
        if ( !log.outputs[0] ) {
            rv = RV_RESOURCE_UNAVAILABLE;
            goto FAIL;
        }
    }

    LOG_MSG( MDL_CORE, SVR_INFO, "log_init: Logger Initialized." );
    return RV_OK;

  FAIL:
    // this will work, since the logger state is valid here
    LOG_MSG( MDL_CORE, SVR_FATAL, "log_init: ERROR Initializing Logger." );
    abort();
}

void log_destroy()
{
    // close logfile, if opened
    if ( log.outputs[0] ) {
        fclose( log.outputs[0] );
        log.outputs[0] = NULL;
    }
}

Severity log_get_severity( Module mdl )
{
    assert( mdl >= 0 && mdl < _MDL_MAX_ );
    return log.max_severity[mdl];
}

void log_set_severity( Module mdl, Severity svr )
{
    assert( mdl >= 0 && mdl < _MDL_MAX_ );
    assert( svr >= 0 && svr < _SVR_MAX_ );
    log.max_severity[mdl] = svr;
}

void log_set_all_severity( Severity svr )
{
    assert( svr >= 0 && svr < _SVR_MAX_ );
    for ( size_t i = 0; i < _MDL_MAX_; ++i )
        log.max_severity[i] = svr;
}

static inline bool log_should_print( Module mdl, Severity svr )
{
    assert( mdl >= 0 && mdl < _MDL_MAX_ && svr >= 0 && svr < _SVR_MAX_ );
    return log.max_severity[mdl] >= svr;
}


static void log_msg_internal(
    Module mdl, Severity svr, const char* file, unsigned int line, const char* func, 
    const char* fmt, va_list ap )
{
#if AED_SYNCHRONIZE_LOG_WRITES
    boost::mutex::scoped_lock lock( mutex );
#endif

    char timestr[16]; // just long enough to fit the dates
    time_t timer;
    time( &timer );
    strftime( timestr, sizeof timestr, "%y%m%d-%H%M%S", localtime( &timer ) );
    timeval highres_timer;
    gettimeofday( &highres_timer, NULL );

    char vstr[1024];
    vstr[0] = '\0';
    vsnprintf( vstr, sizeof vstr, fmt, ap );

    // iterate over all posible outputs, hopefully the compiler will unroll this...
    for ( size_t i = 0; i < MAX_OUTPUTS; i++ ) {

        FILE* f = log.outputs[i];
        if ( f ) {
            // first print the header information
            if (0 == i) {
                fprintf( f, "%u %s %s %zu, %s:%u, %s:\n\t",
                    // thread, module, time(hms), time(us)
                    (unsigned)pthread_self(), mdl_string[mdl], timestr, (size_t)highres_timer.tv_usec,
                    // file, line, func,
                    file, line, func );
            }
            // then print actual message
            fprintf( f, "[%s] %s\n", svr_string[svr], vstr );
            // flush the output in case the program is about to crash
            fflush( f );
        }

    }
}

void _log_msg( Module mdl, Severity svr, 
               const char* func, const char* file, unsigned int line, 
               const char* msg )
{
    _log_var_msg( mdl, svr, func, file, line, "%s", msg );
}

void _log_var_msg( Module mdl, Severity svr, 
                   const char* func, const char* file, unsigned int line, 
                   const char* fmt, ... )
{
    if ( !log_should_print( mdl, svr ) )
        return;

    va_list ap;
    va_start( ap, fmt );
    log_msg_internal(
            mdl, svr, file, line, func, fmt, ap );
    va_end( ap );
}

} /* aed */

// EXAMPLE USAGE:
/*
using namespace aed;

int main(int argc, char* argv[])
{

    log_init( "testfile", stdout );

    // log a simple message
    log_msg( MDL_CORE, SVR_CRITICAL, 
             "this is the first test message." );

    // log a message and a return value, typically used imediately after calling a function
    log_rv_msg( RV_OUT_OF_MEMORY, MDL_CORE, SVR_CRITICAL, 
                "this is the second test message." ); 

    // log a message using prinf args
    log_var_msg( MDL_CORE, SVR_CRITICAL, 
                 "this is the %drd test %s.", 3, "SHIT message" );

    // log a message using prinf args and print a return value.
    log_var_rv_msg( RV_UNSPECIFIED, MDL_CORE, SVR_CRITICAL, 
                    "this is the %dth test %s.", 4, "SB message" );

    printf("\n\n");
    log_destroy();

    return 0;
}
*/

