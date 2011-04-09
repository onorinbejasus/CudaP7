const int MAX_OBJECTS = 128;

uniform sampler2DArray color_tex;
uniform sampler2DArray depth_tex;
uniform float dist;
uniform int index;

const vec3 green = vec3(144.0/255.0, 175.0/255.0, 93.0/255.0);
const int grass_pass = 1;

const float PI = 3.141592653589793238;

/**
 * Returns the direction of the ray going through the given point on
 * the image plane.
 * @param pos The (x,y) position on the image plane, where (-1,-1) is the
 *  bottom left and (1,1) is the top right.
 * @param camera The scene camera.
 * @param mint The min t value for the paremetric equation to be within the
 *  viewing frustum will be stored here.
 * @param maxt The max t value for the paremetric equation to be within the
 *  viewing frustum will be stored here.
 * @return A unit vector pointing from the camera position to the point on
 *  the near clipping plane dictated by pos, for use in the paremetric
 *  equation for a ray.
 */
vec3 compute_eye_ray( vec2 pos, vec3 dir, vec3 up, float fov, float aspect, float near, float far )
{
    float tan_fov = tan( fov / 2.0 );
    // compute direction sans factor of plane distance
    vec3 res = dir + tan_fov * ( aspect * pos.x * cross( dir, up ) + pos.y * up );
    return normalize( res );
}

void main()
{
    vec4 depth_rgba   = texture2DArray( depth_tex, vec3(gl_TexCoord[0].xy, index ) );
    gl_FragColor = texture2DArray( color_tex, vec3(gl_TexCoord[0].xy, index ) );

    float fDepth = depth_rgba.r;  
    float min_dis = dist - 37;
    float max_dis = dist + 37;
    float depth = min_dis + fDepth * (max_dis - min_dis);

    float near = 0.1, far = 1000.0;
    vec3 eye_ray = compute_eye_ray(
        gl_TexCoord[0].xy * 2.0 - vec2(1.0,1.0),
        vec3(0.0,0.0,-1.0), vec3(0.0,1.0,0.0),
        (PI/180.0)*34.33, 1.0, near, far );
    float z = -eye_ray.z * depth;
    z = ( z*(far+near)/(near-far) + (2*far*near)/(far-near) ) / -z;
    z = (z+1.0)/2.0;
    gl_FragDepth = z;

    // HACK
    if ( gl_FragColor.a < 0.1 ) {
        gl_FragDepth = 1.0;
    }
}

