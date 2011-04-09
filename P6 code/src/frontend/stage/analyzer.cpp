
#include "analyzer.hpp"
#include "common/math/matrix.hpp"
#include "common/util/timer.hpp"
#include "frontend/config.hpp"

#include <iostream>
#include <algorithm>
#include <utility>

namespace aed {
namespace frontend {
namespace stage {

// the first is the depth info, the second is the depth sorted ID
typedef std::pair<double, uint32_t> DepthIDPair;
typedef std::vector< DepthIDPair > DepthSortList;


static double get_geometry_depth( const Analyzer::GeometryData& geometry, const Camera &camera )
{
    return dot( geometry.transform.position - camera.get_position(), camera.get_direction() );
}

static bool isFarther( const DepthIDPair &i, const DepthIDPair &j)
{
    return ( i.first > j.first );
}


static bool intersects( Vector3 box1_plane_points[], Vector3 box1_normals[], Vector3 box2_points[] )
{
    // For each point in the box2, compute distances from planes of box1 to the point.
    // If any distance for a point is negative, then the point is outside box1.
    // Assumes box1 normals point inwards.
    for(int point = 0; point < 8; point++)
    {
        bool is_in_box1 = true;
        for(int plane = 0; plane < 6; plane++)
        {
            Vector3 plane_to_point = box2_points[point] - box1_plane_points[plane];
            real_t distance = dot( box1_normals[plane], plane_to_point );
            if(distance < 0)
            { 
                // Negative distance => this point can't be inside box1. Try the next one.
                is_in_box1 = false;
                break;
            }
        }

        // Some point has positive distance to all planes. It must be inside, so these boxes definitely intersect.
        if(is_in_box1)
        {
            return true;
        }
    }

    // No point was inside the box. Boxes might not intersect.
    return false;
}


/**
 * Determine which objects are inside the view frustum and generate a depth
 * sorted list of visible objects.
 * @pre: info must have been reset to a clear state.
 * @post: The only valid entires in each ObjectInfo are object.
 * @post: Sets the camera of the frame info.
 */
static void generate_request_list( FrameInfo& info, const Analyzer::GeometryDataList& geometries, const Camera& camera )
{
    // TODO GET RID OF THIS IT ALLOCATES ON THE HEAP EVERY FRAME
	static DepthSortList depth_sort_list;
    depth_sort_list.clear();

    info.camera = camera;

    // Calculate camera data
	Vector3 cam_pos   = camera.get_position();
	Vector3 cam_dir   = normalize( camera.get_direction() );
	Vector3 cam_up    = normalize( camera.get_up() );
    Vector3 cam_right = cross( cam_dir, cam_up );
 
    real_t tan_fov = tan( camera.get_fov_radians()/2 );
    real_t aspect = camera.get_aspect_ratio();
    real_t near_clip = camera.get_near_clip();
    real_t far_clip = camera.get_far_clip();

    // height and width of the frustum / 2 at the near and far clipping planes
    real_t height_near = tan_fov * camera.get_near_clip();
    real_t width_near = aspect * height_near;
    real_t height_far = tan_fov * camera.get_far_clip();
    real_t width_far = aspect * height_far;

	// Find the vertices, inward normals, and points on the faces of the camera frustum
	
	Vector3 cam_dir_near = cam_pos + (cam_dir * near_clip);
	Vector3 cam_dir_far = cam_pos + (cam_dir * far_clip);
	
	Vector3 frustum_points[8];
	frustum_points[0] = cam_dir_near + (cam_up * -height_near) + (cam_right * -width_near);	// near clip, lower left
	frustum_points[1] = cam_dir_near + (cam_up * -height_near) + (cam_right * width_near);	// near clip, lower right
	frustum_points[2] = cam_dir_near + (cam_up * height_near) + (cam_right * -width_near);	// near clip, upper left
	frustum_points[3] = cam_dir_near + (cam_up * height_near) + (cam_right * width_near);	// near clip, upper right
	frustum_points[4] = cam_dir_far + (cam_up * -height_far) + (cam_right * -width_far);	// far clip, lower left
	frustum_points[5] = cam_dir_far + (cam_up * -height_far) + (cam_right * width_far);	    // far clip, lower right
	frustum_points[6] = cam_dir_far + (cam_up * height_far) + (cam_right * -width_far);	    // far clip, upper left
	frustum_points[7] = cam_dir_far + (cam_up * height_far) + (cam_right * width_far);	    // far clip, upper right

    Vector3 frustum_normals[6];
    frustum_normals[0] = normalize( cross( (frustum_points[2] - frustum_points[0]), (frustum_points[1] - frustum_points[0]) ) );	// near clip
	frustum_normals[1] = normalize( cross( (frustum_points[4] - frustum_points[0]), (frustum_points[2] - frustum_points[0]) ) );	// left face
	frustum_normals[2] = normalize( cross( (frustum_points[5] - frustum_points[4]), (frustum_points[6] - frustum_points[4]) ) );	// far clip
	frustum_normals[3] = normalize( cross( (frustum_points[3] - frustum_points[1]), (frustum_points[5] - frustum_points[1]) ) );	// right face
	frustum_normals[4] = normalize( cross( (frustum_points[6] - frustum_points[2]), (frustum_points[3] - frustum_points[2]) ) );	// top face
	frustum_normals[5] = normalize( cross( (frustum_points[5] - frustum_points[1]), (frustum_points[0] - frustum_points[1]) ) );	// bottom face

    Vector3 frustum_plane_points[6];
    frustum_plane_points[0] = frustum_points[0];
    frustum_plane_points[1] = frustum_points[0];
    frustum_plane_points[2] = frustum_points[5];
    frustum_plane_points[3] = frustum_points[5];
    frustum_plane_points[4] = frustum_points[2];
    frustum_plane_points[5] = frustum_points[0];
    

    // For each object, check if the object is visible. If it is, store its 
    // distance from the camera so that the compositor can render the results 
    // in depth sorted order.
    for( size_t current_id = 0; current_id < geometries.size(); ++current_id ) {

        Vector3 bbox_center = geometries[current_id].transform.position;
        Vector3 bbox = geometries[current_id].bound.size();

        // Find the vertices, inward normals, and points on the faces of the bounding box

		Vector3 box_points[8];
		box_points[0] = Vector3( bbox_center.x + (-0.5 * bbox.x), bbox_center.y + (-0.5 * bbox.y), bbox_center.z + (-0.5 * bbox.z) );	// front lower left
		box_points[1] = Vector3( bbox_center.x + (-0.5 * bbox.x), bbox_center.y + (-0.5 * bbox.y), bbox_center.z + (0.5 * bbox.z) );	// back lower left
		box_points[2] = Vector3( bbox_center.x + (-0.5 * bbox.x), bbox_center.y + (0.5 * bbox.y), bbox_center.z + (-0.5 * bbox.z) );	// front upper left
		box_points[3] = Vector3( bbox_center.x + (-0.5 * bbox.x), bbox_center.y + (0.5 * bbox.y), bbox_center.z + (0.5 * bbox.z) );		// back upper left
		box_points[4] = Vector3( bbox_center.x + (0.5 * bbox.x), bbox_center.y + (-0.5 * bbox.y), bbox_center.z + (-0.5 * bbox.z) );	// front lower right
		box_points[5] = Vector3( bbox_center.x + (0.5 * bbox.x), bbox_center.y + (-0.5 * bbox.y), bbox_center.z + (0.5 * bbox.z) );		// back lower right
		box_points[6] = Vector3( bbox_center.x + (0.5 * bbox.x), bbox_center.y + (0.5 * bbox.y), bbox_center.z + (-0.5 * bbox.z) );		// front upper right
		box_points[7] = Vector3( bbox_center.x + (0.5 * bbox.x), bbox_center.y + (0.5 * bbox.y), bbox_center.z + (0.5 * bbox.z) );		// back upper right

        Vector3 box_normals[6];
		box_normals[0] = normalize( cross( (box_points[2] - box_points[0]), (box_points[1] - box_points[0]) ) );	// near clip
		box_normals[1] = normalize( cross( (box_points[4] - box_points[0]), (box_points[2] - box_points[0]) ) );	// left face
		box_normals[2] = normalize( cross( (box_points[5] - box_points[4]), (box_points[6] - box_points[4]) ) );	// far clip
		box_normals[3] = normalize( cross( (box_points[3] - box_points[1]), (box_points[5] - box_points[1]) ) );	// right face
		box_normals[4] = normalize( cross( (box_points[6] - box_points[2]), (box_points[3] - box_points[2]) ) );	// top face
		box_normals[5] = normalize( cross( (box_points[5] - box_points[1]), (box_points[0] - box_points[1]) ) );	// bottom face

        Vector3 box_plane_points[6];
        box_plane_points[0] = box_points[0];
        box_plane_points[1] = box_points[0];
        box_plane_points[2] = box_points[5];
        box_plane_points[3] = box_points[5];
        box_plane_points[4] = box_points[2];
        box_plane_points[5] = box_points[0];


        // Check if either box is inside camera and if camera is inside box.
        if( true || intersects( frustum_plane_points, frustum_normals, box_points ) || 
            intersects( box_plane_points, box_normals, frustum_points ) )
        {
            DepthIDPair request_id( get_geometry_depth( geometries[current_id], camera ), (uint32_t)current_id );
            depth_sort_list.push_back( request_id );
        }
    }

    // Depth sort the list of visible objects
    sort( depth_sort_list.begin(), depth_sort_list.end(), isFarther );
	for( size_t i = 0; i < depth_sort_list.size(); ++i) {
        ObjectInfo obj;
        obj.object = depth_sort_list[i].second;
        info.objects.push_back( obj );
	}
}


/**
 * @pre: info.object must have already been set.
 * @post: info.clip will be correctly set.
 */
static void calculate_clip( const Analyzer::GeometryData& geometry, const Camera& camera, ObjectInfo& info )
{
	Vector3 cam_pos   = camera.get_position();
	Vector3 cam_dir   = normalize( camera.get_direction() );
	Vector3 cam_up    = normalize( camera.get_up() );
    Vector3 cam_right = cross( cam_dir, cam_up );

    Vector3 center = geometry.transform.position;
    real_t rwidth = (real_t)feconfig.window_width;
    real_t rheight= (real_t)feconfig.window_height;
    real_t tanfov = tan( camera.get_fov_radians() / 2 );

    real_t down = rheight;
    real_t up = 0;
    real_t left = rwidth;
    real_t right = 0;

    Matrix3 view_matrix( cam_right.x, cam_right.y, cam_right.z,
                         cam_up.x   , cam_up.y   , cam_up.z   ,
                         cam_dir.x  , cam_dir.y  , cam_dir.z   );

	const Vector3 bbox = geometry.bound.size();
    for( int i = -1; i <= 1; i+=2 ) {
        for( int j = -1; j <= 1; j+=2 ) {
            for( int k = -1; k <= 1; k+=2 ) {

                Vector3 box_vertex =  center + Vector3( i * bbox.x, j* bbox.y, k* bbox.z );
                Vector3 camera_view_coord = view_matrix * (box_vertex - cam_pos);

                real_t plane_height = 2 * camera_view_coord.z * tanfov;
                real_t plane_width  = plane_height * camera.get_aspect_ratio();

                real_t yintersection = rheight * ( ( camera_view_coord.y / plane_height ) + 0.5);
                real_t xintersection = rwidth *  ( ( camera_view_coord.x / plane_width ) + 0.5 );

                if( xintersection < left ) {
                    left = xintersection;
                }
                if( xintersection > right ) {
                    right = xintersection;
                }
                if( yintersection > up ) {
                    up = yintersection;
                }
                if( yintersection < down ) {
                    down = yintersection;
                }
            }
        }
    }

    // the condition that the object is bigger than screen, so make the screen 
    // size unchanged
    if(down < 0) {
        down = 0;
    }
    if(up > rheight) {
        up = rheight;
    }
    if(left < 0 ) {
        left = 0;
    }
    if(right > rwidth) {
        right = rwidth;
    }

    // XXX this doesn't account for rotation, so it's probably wrong
    info.clip.x = static_cast<uint32_t>( left );
    info.clip.y = static_cast<uint32_t>( down );
    info.clip.width = static_cast<uint32_t>( right - left );
    info.clip.height = static_cast<uint32_t>( up - down );

    info.transform.position = center;
}

static void reset_info( FrameInfo& info )
{
    info.is_rendered = false;
    info.objects.clear();
}

static void set_sending_request(
    BackendRenderingRequestPacket& packet,
    size_t frame, const Analyzer::GeometryData& geometry, const Camera& camera, const ObjectInfo& info, size_t render_order )
{
    // just in case we change it and forget to adjust this
    assert( sizeof packet.header.fe_ident == sizeof(uint64_t) );

    packet.header.type = BRT_RENDER;

    // low bits of backend id matches frontend z-order, high bits are frame index
    packet.header.fe_ident = MAKE_TIMER_ID(render_order, frame, feconfig.frontend_index);

    // type acquired from geometry
    packet.header.renderable = geometry.backend.type_id;
    real_t angle = 0;//geometry->get_geometry_ori( info.object );

    // TODO now all the geometry data is controlled by the front end, so we need to offset the 
    // orientation and translation of the geometry data by rotating and moving the camera
    // HACK for some reason the camera rotation offset didnt achieve the effect we want,
    //      so we send the rotation data to server, and let server do that for us
	Vector3 translation = geometry.transform.position;
    Quaternion rotation = conjugate( Quaternion( Vector3::UnitY, angle ) );
	//std::cout<<"offset info: "<<offset_translation<<std::endl;
	packet.request.transform.camera = camera;

    Camera& cam = packet.request.transform.camera;
	cam.position -= translation;
    cam.position = rotation * cam.position;
    cam.orientation = rotation * cam.orientation;
    
    packet.request.transform.x_offset = info.clip.x;
    packet.request.transform.y_offset = info.clip.y;
    packet.request.transform.clip_width  = info.clip.width;
    packet.request.transform.clip_height = info.clip.height;
    packet.request.transform.window_width  = feconfig.window_width;
    packet.request.transform.window_height = feconfig.window_height;
}

Result Analyzer::initialize()
{
    for (size_t i = 0; i < feconfig.backends.size(); ++i) {
        common::network::Socket* sock;
        Result rv = socketmgr->open_connection( &sock, feconfig.backends[i].hostname.c_str(), feconfig.backends[i].port );
        if ( rv_failed( rv ) ) {
            abort();
        }
        sockets.insert( std::make_pair( i, sock ) );
    }

    for( size_t i = 0; i < frame_infos.size(); ++i ) {
        frame_infos[i].frame = i;
        frame_infos[i].is_rendered = true;
    }
    return RV_OK;
}

void Analyzer::finalize()
{

}

#if 0 // currently unused
static void set_sending_mouse_request( BackendInputNotificationPacket& packet, MouseData& mouseData )
{
    packet.header.type = BRT_FE_INPUT_DATA;

    // Choose the renderable based on the mouse position, then modify the mouse data
    
    int renderable;
    if(mouseData.srcx < 0.5) {
        mouseData.srcx /= 0.5;
        if(mouseData.srcy < 0.5) {
            mouseData.srcy /= 0.5;
            renderable = 2;
        }
        else {
            mouseData.srcy -= 0.5;
            mouseData.srcy /= 0.5;
            renderable = 0;
        }
    } else {
        mouseData.srcx -= 0.5;
        mouseData.srcx /= 0.5;
        if (mouseData.srcy < 0.5) {
            mouseData.srcy /= 0.5;
            renderable = 3;
        } else {
            mouseData.srcy -= 0.5;
            mouseData.srcy /= 0.5;
            renderable = 1;
        }
    }

    // Don't send to a nonexistent renderable
    renderable = std::min( renderable, feconfig.num_geometry-1 );

    packet.header.renderable = renderable;

    packet.inputNotification.mouseData = mouseData;
}
#endif

Result Analyzer::process( const UserInputInfo& uii ) {
   
    // There is no meaningful object id at this stage
    common::util::ManagedTimer analyzerTimer(
            feconfig.analyzerTimerHandle, 
            0);
    
    LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "Analyzer::process");
    FrameInfo& frame = frame_infos[next_frame];
    assert( frame.is_rendered );

    const Camera& camera = uii.camera;
    MouseData mouseData = uii.mouseData;

    // clear old data
    reset_info( frame );


    // Send user mouse input to the backend
    // XXX hard coded to send mouse data to backend 0
#if 0
    if(feconfig.enable_user_input) 
    {
        BackendInputNotificationPacket packet;
        set_sending_mouse_request( packet, mouseData );
        Result rv = socketmgr->send_packet( sockets[0], &packet, sizeof packet );
        if ( rv_failed( rv ) ) {
            LOG_MSG( MDL_FRONT_END, SVR_CRITICAL, "[ERROR] Failed to send input data to backend" );
            return rv;
        }
    }
#endif

    // Get a depth sorted list of all objects in view.
	generate_request_list( frame, geometries, camera );

    // Build a request for each object.
	for( size_t i = 0; i < frame.objects.size(); ++i ) {

        // Start timing for this particular object
        common::util::ManagedTimer analyzerPerObjectTimer(
                feconfig.analyzerObjectTimerHandle, 
                MAKE_TIMER_ID(i,next_frame,feconfig.frontend_index));

        ObjectInfo& obj = frame.objects[i];

		calculate_clip( geometries[obj.object], camera, obj );

        // XXX is this even a good place for this?
        obj.rdata = geometries[obj.object].rdata;

		BackendRenderingRequestPacket packet;
		set_sending_request( packet, next_frame, geometries[obj.object], camera, obj, i );
	
        // Send this packet to the appropriate backend
        int backend_id = geometries[obj.object].backend.backend_id;

        //LOG_VAR_MSG( MDL_FRONT_END, SVR_TRIVIAL, "sending request, backend = %d", backend_id );

		Result rv = socketmgr->send_packet( sockets[backend_id], &packet, sizeof packet );
        LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "Frontend sending render request packet of size %zu to %d", sizeof packet, backend_id );

        if ( rv_failed( rv ) ) {
			LOG_MSG( MDL_FRONT_END, SVR_ERROR, "Failed to send object request" );
            return rv;
		}
	}


    // send frame info to compositor
    common::pipeline::QueuePusher<AnalyzerOutput> handle( queue );
    handle->type = CompositorInput::CI_FrameInfo;
    handle->frame_info = &frame;
    LOG_MSG( MDL_FRONT_END, SVR_NORMAL, "Analyzer sending frame info to compositor" );
    handle.push();

    // increment frame counter
    next_frame++;
    next_frame %= feconfig.frames_in_flight;

    // block until next frame slot has completed rendering and is available,
    // ensures player doesn't send requests too far ahead
    // but exit with error code if aborted

    boost::mutex::scoped_lock lock( frame_mcp->mutex );
    while ( !frame_infos[next_frame].is_rendered && !frame_mcp->is_aborted ) {
        LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "Analyzer blocking until frame %zu finished", next_frame );
        frame_mcp->cvar.wait( lock );
    }

    LOG_VAR_MSG( MDL_FRONT_END, SVR_NORMAL, "Analyzer woke up (frame %zu)", next_frame );

    return frame_mcp->is_aborted ? RV_OPERATION_ABORTED : RV_OK;
}

} // stage
} // frontend
} // aed

