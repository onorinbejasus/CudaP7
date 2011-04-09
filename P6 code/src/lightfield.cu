
#include "lightfield.hpp"
#include "cutil_math.hpp"
#include "cuda_helpers.hh"
#include <stdio.h>
#include <cmath>

#define PI 3.14159265358979f

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define TREE

namespace lightfield {

/* data structures */

struct LFData *camer_info = 0;

static struct LFCamera *gpuCam = 0;
static uint8_t *gpucolor = 0;

static struct LFData *gpuData = 0;
static struct LFData *cpuData = 0;

static float radius = 0;

// Proivdes data for each of the cameras in the lightfield.
// The up vector is not stored in the data, so you MUST at least
// use the code that computes the up vector in your solution.
static __host__ __device__ void get_lightfield_camera( LFCamera* camerap, const LFData* lfdata, size_t camera_index )
{
    // read camera info from lfdata
    float3 neg_dir = lfdata->camera_positions[camera_index];
    camerap->position = neg_dir * lfdata->camera_radius;
    camerap->direction = -neg_dir;
    camerap->fov = lfdata->camera_fov;
    camerap->aspect_ratio = lfdata->camera_aspect;

    // compute up vector the same way as used in the original renders (not in data)
    float3 up_tangent;
    if (fabs(neg_dir.x) < 0.001f && fabs(neg_dir.z) < 0.001f)
        up_tangent = make_float3(-1.0f, 0.0f, 0.0f);
    else if (fabs(neg_dir.y) < 0.001f)
        up_tangent = make_float3(0.0f, 1.0f, 0.0f);
    else
        up_tangent = make_float3(-neg_dir.x, 1.0f/neg_dir.y-neg_dir.y, -neg_dir.z);
    camerap->up = normalize(up_tangent); // NORMALIZE!!!!!!
}

static __host__ __device__ LFCamera get_lightfield_camera(const LFData* lfdata, uint16_t camera_index )
{

    LFCamera camerap;

    // read camera info from lfdata
    float3 neg_dir = lfdata->camera_positions[camera_index];
    camerap.position = neg_dir * lfdata->camera_radius;
    camerap.direction = -neg_dir;
    camerap.fov = lfdata->camera_fov;
    camerap.aspect_ratio = lfdata->camera_aspect;

    // compute up vector the same way as used in the original renders (not in data)
    float3 up_tangent;
    if (fabs(neg_dir.x) < 0.001f && fabs(neg_dir.z) < 0.001f)
        up_tangent = make_float3(-1.0f, 0.0f, 0.0f);
    else if (fabs(neg_dir.y) < 0.001f)
        up_tangent = make_float3(0.0f, 1.0f, 0.0f);
    else
        up_tangent = make_float3(-neg_dir.x, 1.0f/neg_dir.y-neg_dir.y, -neg_dir.z);
    camerap.up = normalize(up_tangent); // NORMALIZE!!!!

    return camerap;
}

__device__ bool find_sphere_t(float3 start, float3 direction, float radius, float2 *t){

    /*find the intersection value of t based on the sphere*/
	/* x^2 + y^2 + z^2 - r^2 = 0 */

	float a, b, c, discrim;		/* the a,b,c and discriminate values for our */
	float2 t0 = make_float2(0.0f);

	/* simplify direction and start point vectors */

	float3 d = direction; float3 p = start;

	/*Calculate the a, b, c and discriminant values for your quadratic equation*/
 	a = dot(d,d); /* d dot d */
 	b = 2 * ( dot(d,p) ); /* 2 * d dot o */
 	c = dot(p,p) - pow(radius, 2);

	discrim = ( pow(b, 2) - (4 * a * c) );

	/*if a is zero, then there is no quadratic possible*/
	if (a == 0)	{
		return false;
	}//end if

	/*if discriminant is < 0, there are two complex roots(no intersection) or one tangential intersecton*/
	if ( discrim <= 0) 	{
		return false;
	}//end if

	/*otherwise, discriminate is > 0 & there are 2 intersection points*/
	else {
		/*find the two roots*/
		t0.x = (-b + sqrt( discrim ) )/ (2 * a);
		t0.y = (-b - sqrt(discrim ) )/ (2 * a);
	}//end else

    *t = t0;

	return true;
}

__global__ void cast_rays(struct LFCamera *cam, uint8_t *color, struct LFData * data, float3 uVec, float3 vVec, size_t width, size_t height, float radius){

    // calculate the unique thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int x = index%width; // unique x pixel
 	int y = index/height;// unique y pixel

    float angle = tan(cam->fov * 0.5);

    float u = (2.0 * x - width)/width ;
    float v = (2.0 * y - height)/height;

    LFCamera cameras[3]; /* three cameras near intersection */

    float3 intersection = make_float3(0.0f);

    // eye ray
    float3 direction = normalize(cam->direction + angle * (uVec * u * cam->aspect_ratio +  v * vVec));

    float2 t = make_float2(0.0f);

    if(find_sphere_t(cam->position, direction, radius, &t)){ // if we intersect with the sphere

        // plug t into implicit sphere equation

        intersection = cam->position + direction * min(t.x, t.y); // find intersection point

        // convert the intersection point to spherical coordinates

        float theta = atan2f(intersection.z, intersection.x);
        theta += (theta < 0) ? 2 * PI : 0; // add PI if less than 0
        float phi = acos(intersection.y / radius);

        if(intersection.y > 0){ // upper hemisphere

            // get the LFindex of the cameras close to my intersection point
            int s = theta/(2*PI) * data->lookup_width - 0.5;
            int t = phi/PI * 2 * data->lookup_height - 0.5;
            int cam_index = s + t * data->lookup_width;

            LFData::LFindex cam_indx; // three closest cameras
            LFData::LFabg bari_Vals; // three baricentric coords

            memcpy(cam_indx, data->indices[cam_index], sizeof(LFData::LFindex) );
            memcpy(bari_Vals, data->abg[cam_index], sizeof(LFData::LFabg) );

//            cam_indx[0] = data->indices[cam_index][0]; // cam1 index
//            cam_indx[1] = data->indices[cam_index][1]; // cam2 index
//            cam_indx[2] = data->indices[cam_index][2]; // cam3 index

            cameras[0] = get_lightfield_camera(data, cam_indx[0]); // get cam 1
            cameras[1] = get_lightfield_camera(data, cam_indx[1]); // get cam 2
            cameras[2] = get_lightfield_camera(data, cam_indx[2]); // get cam 3

            /* field of view of the camera */

            float height_cam = radius * 0.5 * tan(cam->fov * 0.5);
            float width_cam = height_cam * cam->aspect_ratio;

            /* conversion factors */

            float conv_u = (data->camera_tex_width -1) / (2 * width_cam);
            float conv_v = (data->camera_tex_height -1) / (2 * height_cam);

            // new color for pixel
            float4 color_new = make_float4(0.0f);

            for(int i = 0 ;i < 3; i++){

                /* find time of intersection */
                float d = -dot( cam->position, cameras[i].direction)
                                / dot(direction, cameras[i].direction) ;

                /* be sure to add the alpha evn if no intersection*/
                if(d < 0){
                    color_new +=  bari_Vals[i]/255.0 * make_float4(0.0f, 0.0f, 0.0f, 255.0f);
                   continue;
                }

                // uVec and vVec of new cameras

                float3 uVec1 = cross(cameras[i].direction, cameras[i].up);
                float3 vVec1 = cameras[i].up;

                // find intersection point
                float3 plane_int = cam->position + d * direction;

                // find the u, v position
                float u1 = dot(uVec1, plane_int);
                float v1 = dot(vVec1, plane_int);

                u1 = (u1 + width_cam) * conv_u;
                v1 = (v1 + height_cam) * conv_v;

                // offset into the color
                int offset = cam_indx[i] * data->camera_tex_height * data->camera_tex_width;

                u1 = clamp(u1, 0.0, data->camera_tex_width-1);
                v1 = clamp(v1, 0.0, data->camera_tex_height-1);

                int s1 = floor(u1);
                int t1 = floor(v1);

                float4 avg = make_float4(0.0f);

#ifdef BILINEAR

                /* Bilinear interpolation:
                 *   f(x,y) = ( Q11/ ( (x2-x1) * (y2-y1) ) ) * ( (x2-x) * (y2-y) ) +
                *               ( Q21/ ( (x2-x1) * (y2-y1) ) ) * ( (x-x1) * (y2-y) ) +
                *                ( Q12/ ( (x2-x1) * (y2-y1) ) ) * ( (x2-x) * (y-y1) ) +
                *                 ( Q22/ ( (x2-x1) * (y2-y1) ) ) * ( (x1-x) * (y-y1) );
                */

                float x = u1; float y = v1;

                float x1 = floor(x); float x2 = x1 + 1;;

                float y1 = floor(y); float y2 = y1 + 1;

                float denominator = ( (y2-y1) * (x2-x1) );

                float4 Q11 = make_float4(data->color_textures[offset + (int)x1 + (int)y1 * data->camera_tex_width]);
                float4 Q21 = make_float4(data->color_textures[offset + (int)x2 + (int)y1 * data->camera_tex_width]);
                float4 Q12 = make_float4(data->color_textures[offset + (int)x1 + (int)y2 * data->camera_tex_width]);
                float4 Q22 = make_float4(data->color_textures[offset + (int)x2 + (int)y2 * data->camera_tex_width]);

                avg = ( Q11 / denominator * ( (x2 - x) * (y2 - y) )  ) +
                            ( Q21 / denominator * ( (x - x1) * (y2 - y) ) ) +
                            ( Q12 / denominator * ( (x2 - x) * (y - y1) ) ) +
                            ( Q22 / denominator * ( (x - x1) * (y - y1) ) );

                color_new += avg ;// * bari_Vals[i]/255.0;

#else
                /* four closest pixels */
                float4 tex1 = make_float4(data->color_textures[offset + s1 + t1 * data->camera_tex_width]);
                float4 tex2 = make_float4(data->color_textures[offset + (s1+1) + t1 * data->camera_tex_width]);
                float4 tex3 = make_float4(data->color_textures[offset + s1 + (t1+1) * data->camera_tex_width]);
                float4 tex4 = make_float4(data->color_textures[offset + (s1+1) + (t1+1) * data->camera_tex_width]);

                /* take the average */

                avg = (tex1 + tex2 + tex3 + tex4) * 0.25;
                color_new += avg * bari_Vals[i]/255.0;
#endif

            }

#ifdef TREE
            color[index * 4 + 0] = color_new.x;//color_new.x;
            color[index * 4 + 1] = color_new.y;
            color[index * 4 + 2] = color_new.z;
            color[index * 4 + 3] = color_new.w;//color_new.w;

// display indices as colors
#elif defined INDICES

            color[index * 4 + 0] = length(intersection - cameras[0].position) * 30;
            color[index * 4 + 1] = 0;//length(intersection - cameras[1].position) * 64;
            color[index * 4 + 2] = 0;//length(intersection - cameras[2].position) * 64;
            color[index * 4 + 3] = 255;


// display baricentric coords
#elif defined BARI
   baricentric test
            color[index * 4 + 0] = bari_Vals[0];
            color[index * 4 + 1] = bari_Vals[1];
            color[index * 4 + 2] = bari_Vals[2];
            color[index * 4 + 3] = 255;

#endif
        }else{ // lower hemisphere = BLACK

            color[index * 4 + 0] = 0;
            color[index * 4 + 1] = 0;
            color[index * 4 + 2] = 0;
            color[index * 4 + 3] = 0;

        } // end inside else

    }else{ // outside sphere = BLACK

        color[index * 4 + 0] = 0;
        color[index * 4 + 1] = 0;
        color[index * 4 + 2] = 0;
        color[index * 4 + 3] = 0;

    } // end else

} // end cast ray

class MyLightfield : public ILightfield
{
public:
    MyLightfield();
    ~MyLightfield();

    virtual bool initialize( const LFData* data, size_t window_width, size_t window_height );
    virtual void render( uint8_t* color, size_t width, size_t height, const LFCamera* camera );
};

MyLightfield::MyLightfield() {} // default constructor

MyLightfield::~MyLightfield() // default deconstructor
{
    /* clean up */
    cudaFree(gpuData);
    cudaFree(gpuCam);
    cudaFree(gpucolor);
    free(cpuData);
}

bool MyLightfield::initialize( const LFData* data, size_t window_width, size_t window_height )
{
    /* create a pointer for the camera, data, and color */
    cudaMalloc((void**)&gpuCam, sizeof(struct LFCamera));
    cudaMalloc((void**)&gpucolor, 4 * sizeof(uint8_t) * window_height * window_height);
    cudaMalloc((void**)&gpuData, sizeof(struct LFData));

    /* temp cpu struct */
    cpuData = (struct LFData*)malloc(sizeof(struct LFData));

    memcpy(cpuData, data, sizeof(struct LFData));

    /* create GPU pointers to data */
    cudaMalloc((void**)&cpuData->color_textures, sizeof(uint32_t) * cpuData->camera_tex_height * cpuData->camera_tex_width * cpuData->num_cameras);
    cudaMemcpy(cpuData->color_textures, data->color_textures, sizeof(uint32_t) * cpuData->camera_tex_height * cpuData->camera_tex_width * cpuData->num_cameras, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cpuData->indices, sizeof(LFData::LFindex) * cpuData->lookup_height * cpuData->lookup_width);
    cudaMemcpy(cpuData->indices, data->indices, sizeof(LFData::LFindex) * cpuData->lookup_height * cpuData->lookup_width, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cpuData->abg, sizeof(LFData::LFabg) * cpuData->lookup_height * cpuData->lookup_width);
    cudaMemcpy(cpuData->abg, data->abg, sizeof(LFData::LFabg) * cpuData->lookup_height * cpuData->lookup_width, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cpuData->camera_positions, sizeof(float3) * cpuData->num_cameras);
    cudaMemcpy(cpuData->camera_positions, data->camera_positions, sizeof(float3) * cpuData->num_cameras, cudaMemcpyHostToDevice);

    /* copy temp struct to gpu */
    cudaMemcpy(gpuData, cpuData, sizeof(struct LFData), cudaMemcpyHostToDevice);

    /* copy over the radius for easy use */
    radius = cpuData->camera_radius;

    return true;
}

void MyLightfield::render( uint8_t* color, size_t width, size_t height, const LFCamera* camera )
{
    /* copy over the viewers camera into gpu memory */
    cudaMemcpy(gpuCam, camera, sizeof(struct LFCamera), cudaMemcpyHostToDevice);

    /* constants for kernel */

	const int threadsPerBlock = 512 ;

	int totalThreads = width * height;
	int nBlocks = totalThreads / threadsPerBlock;
	nBlocks += ((totalThreads % threadsPerBlock) > 0) ? 1 : 0;

    float3 uVec = cross(camera->direction, camera->up);
    float3 vVec = camera->up;

    /* launch kernel */
    cast_rays<<<nBlocks, threadsPerBlock>>>(gpuCam, gpucolor, gpuData, uVec, vVec, width, height, radius);

    /* copy back data */
    cudaMemcpy(color, gpucolor, 4 * sizeof(uint8_t) * totalThreads, cudaMemcpyDeviceToHost);

}

ILightfield* create_lightfield()
{
    return new MyLightfield();
}

}
