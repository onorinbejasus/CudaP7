/** @file log.hpp
 *
 *  A module for logging/debug printing. Use this for all printed messages.
 *  DO NOT USE printf, etc, directly.
 */

#ifndef _AED_LOG_HPP_
#define _AED_LOG_HPP_

#include "common/result.hpp"

namespace aed {

// XXX improve these
enum Module {
    MDL_CORE,           // everything in the common/core/ folder.
    MDL_MATH,           // everything in the math/ folder.
    MDL_RENDER,         // generic rendering code, some things in render/ folder.
    MDL_TEXTURE,
    MDL_PIPELINE,
    MDL_BACK_END,       // everything in BACK END.
	MDL_FRONT_END,
    MDL_NETWORK,
    MDL_COMPRESSION,    // (de)compression error.
    // add more as needed, also add string to source file...
    _MDL_MAX_           // unused, marks max value for array
};

enum Severity {
    SVR_FATAL,          // irrecoverable error, must abort process immediately
    SVR_ERROR,          // recoverable error, must abort current activity
    SVR_WARNING,        // potential problem, but can continue current activity
    SVR_INFO,           // normal but important program messages, like initialization
    SVR_DEBUG1,         // verbose debugging info
    SVR_DEBUG2,
    SVR_DEBUG3,
    _SVR_MAX_
};

// DEPRECATED. left here since a lot of code uses the old severity types
#define SVR_CRITICAL SVR_ERROR
#define SVR_NORMAL SVR_DEBUG2
#define SVR_TRIVIAL SVR_DEBUG3
#define SVR_INIT SVR_INFO

/**
 * Initializes the logger. Logging will occur to all non-null locations.
 * Initializes severity levels to SVR_CRITICAL | SVR_INIT.
 * @param logfile.  The path of a file to which to log output. May be null, in
 *                  which case nothing is logged to this stream.
 * @param print_stream. A second stream (usually stderr or stdout) to which to
 *                      dump output. May be null, in which case nothing is 
 *                      logged to this stream.
 */
Result log_init( const char* logfile, FILE* print_stream );

/**
 * Finalizes the logger and shuts it down.
 */
void log_destroy();

/**
 * Query the current severity level for the given module.
 */
Severity log_get_severity( Module mdl );

/**
 * Set the severity level for the given module.
 */
void log_set_severity( Module mdl, Severity svr );

/**
 * Set the severity level for all modules.
 */
void log_set_all_severity( Severity svr );

/**
 * Log an unformatted message. The module and a timestamp will also be printed.
 */
#define LOG_MSG( mdl, svr, msg )                                    \
    _log_msg( mdl, svr, __func__, __FILE__, __LINE__, msg )

/**
 * Log an unformatted message. The return value, module and a timestamp will
 * also be printed.
 * @return The rv that was passed in.
 */
#define LOG_RV_MSG( rv, mdl, svr, msg )                             \
    _log_rv_msg( rv, mdl, svr, __func__, __FILE__, __LINE__, msg )

/**
 * Log a formatted message. Format string behaves like the printf family.
 * The module and a timestamp will also be printed.
 */
#define LOG_VAR_MSG( mdl, svr, fmt, args... )                       \
    _log_var_msg( mdl, svr, __func__, __FILE__, __LINE__, fmt, ## args )

/**
 * Log a formatted message. Format string behaves like the printf family.
 * The return value, module, and a timestamp will also be printed.
 * @return The rv that was passed in.
 */
#define LOG_VAR_RV_MSG( rv, mdl, svr, fmt, args... )                \
    _log_var_rv_msg( rv, mdl, svr, __func__, __FILE__, __LINE__, fmt, ## args )

void _log_msg( 
        Module mdl, Severity svr, 
        const char* func, const char* file, unsigned int line, 
        const char* msg );

void _log_var_msg(
        Module mdl, Severity svr, 
        const char* func, const char* file, unsigned int line, 
        const char* fmt, ... ) __attribute__((format (printf, 6, 7)));

} /* aed */

#endif /* _AED_LOG_HPP_ */

