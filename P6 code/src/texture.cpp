
#include "texture.hpp"
#include "common/util/imageio.hpp"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <vector>

namespace lightfield {

#define PI 3.14159265358979

template<typename T>
static inline T clamp_val( T val, T min, T max )
{
	return std::min( std::max( val, min ), max );
}

// load the index data
static LFData::LFindex* load_indices( const char* filename, size_t width, size_t height )
{
    printf( "Loading index table..." );
    fflush( stdout );

	LFData::LFindex* result;
	size_t num_tex = width * height;
	size_t size = num_tex * sizeof *result;

	result = (LFData::LFindex*) malloc( size );
	memset( result, 0, size );

	// each line is one pixel and contains 3 space-delimited ints
    std::ifstream ifs;
    ifs.open( filename, std::ios::in );
    if( !ifs.is_open() ) {
        printf( " ERROR opening file.\n" );
        free( result );
        return NULL;
    }
	for ( size_t i = 0; i < num_tex; i++ ) {
		uint16_t r, g, b;
		ifs >> r >> g >> b;
		if ( ifs.bad() ) {
            printf( " ERROR reading data.\n" );
            free( result );
            return NULL;
        }
		result[i][0] = r;
		result[i][1] = g;
		result[i][2] = b;
	}

    printf( "done.\n" );
	return result;
}

static LFData::LFabg* load_abg( const char* filename, size_t width, size_t height )
{
    printf( "Loading barycentric coordinate table..." );
    fflush( stdout );

	LFData::LFabg* result;
	size_t num_tex = width * height;
	size_t size = num_tex * sizeof *result;

	result = (LFData::LFabg*) malloc( size );
	memset( result, 0, size );

	// each line is one pixel and contains 3 space-delimited floats
    std::ifstream ifs;
    ifs.open( filename, std::ios::in );
    if( !ifs.is_open() ) {
        printf( " ERROR opening file.\n" );
        free( result );
        return NULL;
    }
	for ( size_t i = 0; i < num_tex; i++ ) {
		double alpha, beta, gamma;
		ifs >> alpha >> beta >> gamma;
		if ( ifs.bad() ) {
            printf( " ERROR reading data.\n" );
            free( result );
            return NULL;
        }
		result[i][0] = (uint8_t)clamp_val(alpha * 255.0, 0.0, 255.0);
		result[i][1] = (uint8_t)clamp_val(beta  * 255.0, 0.0, 255.0);
		result[i][2] = (uint8_t)clamp_val(gamma * 255.0, 0.0, 255.0);
	}

    printf( "done.\n" );
	return result;
}

static float3* load_pos( const char* filename, size_t num_cameras )
{
    printf( "Loading camera position table..." );
    fflush( stdout );

	float3* result;
	size_t size = num_cameras * sizeof *result;

	result = (float3*) malloc( size );
	memset( result, 0, size );

	// each line is one camera and contains 3 space-delimited floats
    std::ifstream ifs;
    ifs.open( filename, std::ios::in );
    if( !ifs.is_open() ) {
        printf( " ERROR opening file.\n" );
        free( result );
        return NULL;
    }
	for ( size_t i = 0; i < num_cameras; i++ ) {
		float x, y, z;
		ifs >> x >> y >> z;
		if ( ifs.bad() ) {
            printf( " ERROR reading data.\n" );
            free( result );
            return NULL;
        }
		result[i] = make_float3( x, y, z );
	}

    printf( "done.\n" );
	return result;
}

static uint32_t* load_textures( size_t* widthp, size_t* heightp, const char* path_format, size_t num_textures )
{
    printf( "Loading camera textures...\n" );
    fflush( stdout );

	uint32_t* result = 0;
	int width, height;
	int num_pixels;

	for ( size_t k = 0; k < num_textures; k++ ) {
		if ( k % 10 == 0 || k == num_textures - 1 ) {
			printf( "\rLoading texture %04zu...", k+1 );
			fflush( stdout );
		}

		char file[2048];
		snprintf( file, sizeof file, path_format, k+1 );

		// load the image
		int curr_width, curr_height;
		uint8_t* texture = aed::common::util::imageio_load_image( file, &curr_width, &curr_height );
        if ( !texture ) {
            printf( "ERROR opening texture '%s'.\n", file );
            free( result );
            return NULL;
        }

		// determine size (or verify size is the same for subsequent images)
		if ( k == 0 ) {
			width = curr_width;
			height = curr_height;
			num_pixels = curr_width * curr_height;
			result = (uint32_t*) malloc( num_textures * num_pixels * sizeof *result );
		} else {
			if (!( width == curr_width && height == curr_height )) {
                printf( "ERROR: texture is not same size: '%s'.\n", file );
                free( result );
                return NULL;
            }
		}

		// copy data into one giant array and free image
		memcpy( &result[k * num_pixels], texture, num_pixels * sizeof *result );
		free( texture );
	}

    printf( "done.\n" );

	*widthp = width;
	*heightp = height;
	return result;
}

static const char DEFAULT_ROOT_DATA_DIR[] = "/afs/cs.cmu.edu/academic/class/15668-s11/p5/data";
static const char COLOR_TEXTURE_PATH_FORMAT[] = "%s/images/%zu/chestnut%s.png";
static const char INDEX_LOOKUP_FILENAME[] = "%s/lookup/%zu/pixel_lookup.txt";
static const char ABG_LOOKUP_FILENAME[] = "%s/lookup/%zu/abg_lookup.txt";
static const char POS_LOOKUP_FILENAME[] = "%s/lookup/%zu/vertices_array_xyz.txt";
static const float LF_CAMERA_RADIUS = 37.0f;
static const float LF_CAMERA_FOV = (float)(37.0 / 180.0 * PI);
static const float LF_CAMERA_ASPECT = 1.0f;
static const size_t LOOKUP_SIZE = 2048;
static const size_t NUM_CAMERAS[2] = { 545, 2113 };

// returns true on success, false on failure
bool load_lightfield( LFData* result, const char* root_data_dir, LFNumCameras num_cameras, LFTextureSize texture_size )
{
    // default to project data dir
    if ( root_data_dir == NULL ) {
        root_data_dir = DEFAULT_ROOT_DATA_DIR;
    }

    result->camera_radius = LF_CAMERA_RADIUS;
    result->camera_fov = LF_CAMERA_FOV;
    result->camera_aspect = LF_CAMERA_ASPECT;

    result->lookup_width = LOOKUP_SIZE;
    result->lookup_height = LOOKUP_SIZE / 2;

    result->num_cameras = NUM_CAMERAS[num_cameras];

    result->color_textures = 0;
    result->indices = 0;
    result->abg	= 0;
    result->camera_positions = 0;

    char color_tex_format[2048];
    char index_filename[2048];
    char abg_filename[2048];
    char pos_filename[2048];

    size_t sz = texture_size == LFTS_256 ? 256 : 128;
    snprintf( color_tex_format, sizeof color_tex_format, COLOR_TEXTURE_PATH_FORMAT, root_data_dir, sz, "%zu" );
    snprintf( index_filename, sizeof index_filename, INDEX_LOOKUP_FILENAME, root_data_dir, result->num_cameras );
    snprintf( abg_filename, sizeof abg_filename, ABG_LOOKUP_FILENAME, root_data_dir, result->num_cameras );
    snprintf( pos_filename, sizeof pos_filename, POS_LOOKUP_FILENAME, root_data_dir, result->num_cameras );

    result->color_textures = load_textures( &result->camera_tex_width, &result->camera_tex_height, color_tex_format, result->num_cameras );
    if ( result->color_textures == 0 ) goto FAIL;
    result->indices = load_indices( index_filename, result->lookup_width, result->lookup_height );
    if ( result->indices == 0 ) goto FAIL;
    result->abg	= load_abg( abg_filename, result->lookup_width, result->lookup_height );
    if ( result->abg == 0 ) goto FAIL;
    result->camera_positions = load_pos( pos_filename, result->num_cameras );
    if ( result->camera_positions == 0 ) goto FAIL;

	fflush( stdout );
    return true;

  FAIL:
    free( result->color_textures );
    free( result->indices );
    free( result->abg );
    free( result->camera_positions );
	printf( "FAILED to load lightfield data. Did you forget to run 'aklog cs.cmu.edu' or to pass in your custom data directory?\n" );
	fflush( stdout );
    return false;
}

}

