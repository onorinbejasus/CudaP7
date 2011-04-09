#ifndef AED_PIPELINE_PIPELINE_HPP
#define AED_PIPELINE_PIPELINE_HPP

#include "common/log.hpp"
#include "common/result.hpp"
#include "common/pipeline/queue.hpp"

#include <boost/bind.hpp>
#include <boost/thread.hpp>

namespace aed {
namespace common {
namespace pipeline {

template <typename Input>
class IPipelineStage
{
public:
    virtual ~IPipelineStage() { }
    virtual boost::thread* run_on_new_thread() = 0;
};

template <typename Stage, typename Input>
class PipelineStage : public IPipelineStage<Input>
{
public:
    PipelineStage( Stage* stage, PipelineQueue<Input>* input );
    virtual ~PipelineStage();

    virtual boost::thread* run_on_new_thread();

private: 
    PipelineStage(const PipelineStage&);
    void operator=(const PipelineStage&);

    void run();

    Stage* _stage;
    CircularQueue* _input;
};

template <typename Stage, typename Input>
IPipelineStage<Input>* create_stage( Stage* stage, PipelineQueue<Input>* input )
{
    return new PipelineStage<Stage, Input>( stage, input );
}

template <typename Stage, typename Input>
PipelineStage<Stage,Input>::PipelineStage( Stage* stage, PipelineQueue<Input>* input )
  : _stage( stage ),
    _input( 0 )
{
    assert( stage && input );

    // extract queue
    _input = (CircularQueue*) input;
    // make sure the queue makes sense
    assert( _input->elem_size() == sizeof( Input ) );
}

template <typename Stage, typename Input>
PipelineStage<Stage,Input>::~PipelineStage()
{

}

template <typename Stage, typename Input>
boost::thread* PipelineStage<Stage,Input>::run_on_new_thread()
{
    return new boost::thread(boost::bind(&PipelineStage<Stage, Input>::run, this));
}

template <typename Stage, typename Input>
void PipelineStage<Stage,Input>::run()
{
    LOG_VAR_MSG( MDL_PIPELINE, SVR_INFO, "starting pipeline stage '%s'", typeid( Stage ).name() );

    try {
        Result rv;

        rv = _stage->initialize();
        if ( rv_failed( rv ) ) {
            LOG_VAR_MSG( MDL_PIPELINE, SVR_ERROR, "failed initializing pipeline stage '%s'", typeid( Stage ).name() );
            return;
        }

        try {
            while ( true ) {
                void* inp;

                // pop a new request
                rv = _input->block_until_pop( &inp );
                if ( rv_failed( rv ) ) {
                    LOG_VAR_MSG( MDL_PIPELINE, SVR_ERROR, "pipeline '%s' failed to POP a request", typeid( Stage ).name() );
                    break;
                } else {
                    LOG_VAR_MSG( MDL_PIPELINE, SVR_DEBUG2, "'%s' has popped a new item %p", typeid( Stage ).name(), inp );
                }

                // process
                Result proc_rv;
                try {
                    proc_rv = _stage->process( *((Input*)inp) );
                } catch ( ... ) {
                    LOG_VAR_MSG( MDL_PIPELINE, SVR_ERROR, "exception during 'process' for '%s'", typeid( Stage ).name() );
                    proc_rv = RV_UNSPECIFIED;
                }

                // chech return status of process() 
                if ( rv_failed( proc_rv ) ) {
                    LOG_VAR_MSG( MDL_PIPELINE, SVR_ERROR, "pipeline '%s' failed to PROCESS a request", typeid( Stage ).name() );
                    break;
                } else {
                    LOG_VAR_MSG( MDL_PIPELINE, SVR_DEBUG2, "'%s' has finished processing item %p", typeid( Stage ).name(), inp );
                }

                // always release, even if process fails
                ((Input*)inp)->~Input();
                rv = _input->release( inp );

                if ( rv_failed( rv ) ) {
                    LOG_VAR_MSG( MDL_PIPELINE, SVR_ERROR, "pipeline '%s' failed to RELEASE a request", typeid( Stage ).name() );
                    break;
                }
            }

            LOG_VAR_MSG( MDL_PIPELINE, SVR_INFO, "pipeline loop exited normally for '%s'", typeid( Stage ).name() );

        } catch ( std::runtime_error ) { 
            LOG_VAR_MSG( MDL_PIPELINE, SVR_INFO, "pipeline loop exited with exception for '%s'", typeid( Stage ).name() );
        }

        _stage->finalize();
    } catch ( ... ) {
        LOG_VAR_MSG( MDL_PIPELINE, SVR_ERROR, "failure during startup/shutdown of stage '%s'", typeid( Stage ).name() );
    }

    LOG_VAR_MSG( MDL_PIPELINE, SVR_INFO, "closing pipeline stage '%s'", typeid( Stage ).name() );
}

} // pipeline
} // common
} // aed

#endif /* AED_PIPELINE_PIPELINE_HPP */


