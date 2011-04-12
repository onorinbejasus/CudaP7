#ifndef AED_FRONTEND_STAGE_ANALYZER_HPP
#define AED_FRONTEND_STAGE_ANALYZER_HPP

#include "common/math/camera.hpp"
#include "common/math/geometry.hpp"
#include "common/math/box.hpp"
#include "common/network/socket.hpp"
#include "common/pipeline/queue.hpp"
#include "frontend/player.hpp"
#include "frontend/config.hpp"
#include <map>

namespace aed {
namespace frontend {
namespace stage {

typedef UserInputInfo AnalyzerInput;
typedef CompositorInput AnalyzerOutput;


class Analyzer
{
public:

    struct Backend {
        size_t backend_id;
        size_t type_id;
    };

    struct GeometryData {
        aed::Transform transform;
        BoundingBox3 bound;
        Backend backend;
        ObjectRenderData rdata;
    };

    typedef std::vector< GeometryData > GeometryDataList;

    Analyzer(
        common::network::SocketManager* socketmgr,
        const GeometryDataList& geometries,
        MutexCvarPair* frame_mcp,
        common::pipeline::PipelineQueue<AnalyzerOutput>& queue
      )
      : frame_infos( MAX_FRAMES_IN_FLIGHT ),
        socketmgr( socketmgr ),
        geometries( geometries ),
        next_frame( 0u ),
        frame_mcp( frame_mcp ),
        queue( queue ),
        sockets( sockets )
        {}
    ~Analyzer() {}

    Result initialize();
    void finalize();

    Result process( const AnalyzerInput& request );

private:

    std::vector<FrameInfo> frame_infos;

    common::network::SocketManager* socketmgr;

    GeometryDataList geometries;

    size_t next_frame;
    MutexCvarPair* frame_mcp;

    common::pipeline::PipelineQueue<AnalyzerOutput>& queue;

    // map from backend ids to sockets
    std::map< size_t, common::network::Socket* > sockets;
};

} // stage
} /* frontend */
} /* aed */

#endif /* AED_ANALYZER_HPP */

