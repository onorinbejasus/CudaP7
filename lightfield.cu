#include "lightfield.hpp"
#include "cutil_math.hpp"
#include "cuda_helpers.hh"
#include <stdio.h>

#define PI 3.14159265358979f

namespace lightfield {

// Local CPU version of LFData
static struct LFData *local = 0;

// GPU versions of camera, data and colors
static struct LFData* cudaData = 0;
static struct LFCamera* cudaCamera = 0;
static uint8_t *cudaColor = 0;

// The radius of the global camera
static float cameraRadius = 0;

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
    camerap->up = normalize(up_tangent);
}

class MyLightfield : public ILightfield
{
public:
    MyLightfield();
    ~MyLightfield();

    virtual bool initialize( const LFData* data, size_t window_width, size_t window_height );
    virtual void render( uint8_t* color, size_t width, size_t height, const LFCamera* camera );
};

MyLightfield::MyLightfield()
{
    // TODO implement
}

MyLightfield::~MyLightfield()
{
    // TODO implement
	cudaFree(cudaData);
	cudaFree(cudaCamera);
	cudaFree(cudaColor);
	free(local);
}

bool MyLightfield::initialize( const LFData* data, size_t window_width, size_t window_height )
{
    // TODO implement
	cameraRadius = data->camera_radius;

	// Allocate for LFData
	local = (struct LFData*)malloc(sizeof(struct LFData));
	memcpy(local, data, sizeof(struct LFData));

	// Allocate arrays in LFData
	cudaMalloc((void**)&local->color_textures, sizeof(uint32_t) * local->camera_tex_width * local->camera_tex_height * local->num_cameras);
	cudaMemcpy(local->color_textures, data->color_textures, sizeof(uint32_t)*local->camera_tex_width*local->camera_tex_height*local->num_cameras, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&local->indices, sizeof(LFData::LFindex) * local->lookup_width * local->lookup_height);
	cudaMemcpy(local->indices, data->indices, sizeof(LFData::LFindex) * local->lookup_width * local->lookup_height, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&local->abg, sizeof(LFData::LFabg) * local->lookup_width * local->lookup_height);
	cudaMemcpy(local->abg, data->abg, sizeof(LFData::LFabg) * local->lookup_width * local->lookup_height, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&local->camera_positions, sizeof(float3) * local->num_cameras);
	cudaMemcpy(local->camera_positions, data->camera_positions, sizeof(float3) * local->num_cameras, cudaMemcpyHostToDevice);

	// Allocate LFData for GPU memory and do a memcpy
	cudaMalloc((void**)&cudaData, sizeof(struct LFData));
	cudaMemcpy(cudaData, local, sizeof(struct LFData), cudaMemcpyHostToDevice);

	// Allocate for world camera
	cudaMalloc((void**)&cudaCamera, sizeof(struct LFCamera));

	// Allocate for pixel colors
	cudaMalloc((void**)&cudaColor, sizeof(uint8_t) * 4 * window_width * window_height);

    return true;
}

/**
 * Takes a ray, a source, location and a radius to compute of the ray intersects a sphere with radius
 * rad. If it does not intersect the sphere, the function returns false. If it does intersect the sphere
 * it returns true and writes the time of intersection to t.
 */
__device__ bool calcSphereIntersect(float3 ray, float3 source, float rad, float *t)
{
	float A, B, C, discriminant;

	A = dot(ray, ray);
	B = 2.0 * dot(ray, source);
	C = dot(source, source) - rad*rad;

	discriminant = B*B - 4*A*C;

	if(discriminant < 0.0)
	{
		return false;
	}

	float sqrtDis = sqrt(discriminant);
	float t1 = (-B - sqrtDis)/(2.0*A);

	*t = t1;

	return true;
}

__global__ void castRays(LFData* data, const LFCamera* camera, uint8_t *color, size_t width, size_t height, float3 uVec, float3 vVec, float rad)
{
	// Calculate the thread index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// To make later computations easier
	size_t baseIndex = index * 4;

	// Final color
	float4 final_color = make_float4(0.0f);
	color[baseIndex + 0] = final_color.x;
	color[baseIndex + 1] = final_color.y;
	color[baseIndex + 2] = final_color.z;
	color[baseIndex + 3] = final_color.w;
			
	// Compute the current pixel (x,y)
	size_t x = index % width;
	size_t y = index / height;

	float angle = tan((camera->fov) * 0.5);
	float u = (2.0*x - width)/width;
	float v = (2.0*y - height)/height;
	
	// Indicies which give us three values to grab from camera array
	LFData::LFindex cameraIndicies;
	LFData::LFabg baryCoords;

	// Array of three cameras
	LFCamera cameras[3];

	// Point of intersection
	float3 intersect = make_float3(0.0f);

	// Time of intersection
	float t;

	// Eyeray from camera to scene
	float3 eyeray = normalize(camera->direction + angle*(uVec*u*camera->aspect_ratio + v*vVec));

	if( calcSphereIntersect(eyeray, camera->position, rad, &t) )
	{
		// Ray intersects tree sphere
		intersect = camera->position + t*eyeray;

		// Calculate the spherical coordinates
		float phi = acos(intersect.y/rad);
		float theta = atan2f(intersect.z, intersect.x);
		if(theta < 0)
			theta += 2*PI;

		if(intersect.y > 0.0)
		{
			// Upper hemisphere

			// Find cameras
			int s = theta/(2*PI) * data->lookup_width - 0.5;
			int t = phi/PI*2 * data->lookup_height - 0.5;
			int dataIndex = s + t*data->lookup_width;

			// Calculate the camera indicies
			cameraIndicies[0] = data->indices[dataIndex][0];
			cameraIndicies[1] = data->indices[dataIndex][1];
			cameraIndicies[2] = data->indices[dataIndex][2];

			// Calculate the barycentric coordinates
			baryCoords[0] = data->abg[dataIndex][0];
			baryCoords[1] = data->abg[dataIndex][1];
			baryCoords[2] = data->abg[dataIndex][2];

			// Calculate the up vectors for each camera
			get_lightfield_camera(&cameras[0], data, cameraIndicies[0]);
			get_lightfield_camera(&cameras[1], data, cameraIndicies[1]);
			get_lightfield_camera(&cameras[2], data, cameraIndicies[2]);

			float cameraHeight = rad * 0.5 * tan(camera->fov * 0.5);
			float cameraWidth = cameraHeight * camera->aspect_ratio;

			float uConversion = (data->camera_tex_width - 1) / (2 * cameraWidth);
			float vConversion = (data->camera_tex_height - 1) / (2 * cameraHeight);

			// For the three closest cameras, cast more rays...
			for(int i = 0; i < 3; i++)
			{
				LFCamera curr = cameras[i];

				// Calculate the time of intersection
				float timeOfIntersection = -dot(camera->position, curr.direction)/dot(eyeray, curr.direction);
				if(timeOfIntersection < 0)
				{
					final_color += baryCoords[i]/255.0 * make_float4(0.0f, 0.0f, 0.0f, 255.0f);
					continue;
				}

				float3 newUVec = cross(curr.direction, curr.up);
				float3 newVVec = curr.up;

				float3 planeIntersection = camera->position + eyeray * timeOfIntersection;

				float newU = dot(newUVec, planeIntersection);
				float newV = dot(newVVec, planeIntersection);

				newU = (newU + cameraWidth) * uConversion;
				newV = (newV + cameraHeight) * vConversion;

				int offset = cameraIndicies[i] * data->camera_tex_height * data->camera_tex_width;

				newU = clamp(newU, 0.0, data->camera_tex_width - 1);
				newV = clamp(newV, 0.0, data->camera_tex_height - 1);

				int newS = floor(newU);
				int newT = floor(newV);

				float4 currColor = make_float4(0.0f);

				float4 tex1 = make_float4(data->color_textures[offset + newS + newT * data->camera_tex_width]);
				float4 tex2 = make_float4(data->color_textures[offset + (newS+1) + newT * data->camera_tex_width]);
				float4 tex3 = make_float4(data->color_textures[offset + newS + (newT+1) * data->camera_tex_width]);
				float4 tex4 = make_float4(data->color_textures[offset + (newS+1) + (newT+1) * data->camera_tex_width]);
	
				// Interpolate the texture colors
				currColor = (tex1 + tex2 + tex3 + tex4)*0.25;

				// Weight the camera's texture colors based on the barycentric coordinates and add that to final_color
				final_color += currColor * baryCoords[i]/255.0;
			}

			// Write the final_color values to the pixel colors
			color[baseIndex + 0] = final_color.x;
			color[baseIndex + 1] = final_color.y;
			color[baseIndex + 2] = final_color.z;
			color[baseIndex + 3] = final_color.w;
		}
	}
}

void MyLightfield::render( uint8_t* color, size_t width, size_t height, const LFCamera* camera )
{
    // TODO implement
	cudaMemcpy(cudaCamera, camera, sizeof(struct LFCamera), cudaMemcpyHostToDevice);

	// Cast ray from each pixel
	float3 uVec = cross(camera->direction, camera->up);
	float3 vVec = camera->up;

	const int threadsPerBlock = 128;
	int numThreads = width * height;
	int nBlocks = numThreads / threadsPerBlock;
	nBlocks += ((numThreads % threadsPerBlock) > 0) ? 1 : 0;

	// Run kernel and write the colors returned by the kernel to the color buffer
	castRays<<<nBlocks, threadsPerBlock>>>(cudaData, cudaCamera, cudaColor, width, height, uVec, vVec, cameraRadius);
	cudaMemcpy(color, cudaColor, sizeof(uint8_t) * 4 * width * height, cudaMemcpyDeviceToHost);
}

ILightfield* create_lightfield()
{
    return new MyLightfield();
}

}



