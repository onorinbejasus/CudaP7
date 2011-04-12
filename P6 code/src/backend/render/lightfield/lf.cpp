
#include "lf.hpp"

namespace aed {
namespace backend {
namespace render {
namespace lightfield {

static const unsigned int MAX_FW_WINDOW_WIDTH  = 512;
static const unsigned int MAX_FW_WINDOW_HEIGHT = 512;

static float3 v32f3( const Vector3& v ) { return make_float3( (float)v.x, (float)v.y, (float)v.z ); }

Result LightfieldRenderAlgorithm::initialize( size_t )
{
	color_data = new uint8_t[MAX_FW_WINDOW_WIDTH*MAX_FW_WINDOW_HEIGHT*4];
	memset( color_data, 0, MAX_FW_WINDOW_WIDTH*MAX_FW_WINDOW_HEIGHT*4 );

	lf = ::lightfield::create_lightfield();
	if ( lf->initialize( &lfdata, MAX_FW_WINDOW_WIDTH, MAX_FW_WINDOW_HEIGHT ) ) {
		return RV_OK;
	} else {
		return RV_UNSPECIFIED;
	}
}

void LightfieldRenderAlgorithm::destroy()
{
	delete [] color_data;
}

// render a frame. 
Result LightfieldRenderAlgorithm::render( uint8_t* color_buffer, uint8_t* depth_buffer, const TransformData& transform )
{
    if ( transform.clip_width > MAX_FW_WINDOW_WIDTH || transform.clip_height > MAX_FW_WINDOW_HEIGHT ) {
		return RV_OK;
    }

	::lightfield::LFCamera cam;
    cam.position	= v32f3( transform.camera.position );
    cam.direction	= v32f3( transform.camera.get_direction() );
    cam.up			= v32f3( transform.camera.get_up() );
    cam.fov			= transform.camera.fov;
    cam.aspect_ratio = transform.camera.aspect;

	lf->render( color_data, MAX_FW_WINDOW_WIDTH, MAX_FW_WINDOW_HEIGHT, &cam );

	for ( size_t y = 0; y < transform.clip_height; y++ ) {
		memcpy(
			&color_buffer[4 * y * transform.clip_width],
			&color_data[4 * ((y+transform.y_offset)*MAX_FW_WINDOW_WIDTH + transform.x_offset)],
			4 * transform.clip_width
		  );
	}

	memset( depth_buffer, 0, 1 * transform.clip_width * transform.clip_height );

    return RV_OK;
}

}
}
}
}

