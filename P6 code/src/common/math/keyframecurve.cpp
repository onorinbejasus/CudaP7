#include "keyframecurve.hpp"
#include <SDL/SDL.h>
#include <iomanip>

namespace aed {

#ifdef  DEBUG_KEY_FRAME_CURVE
Vector3 g_x_vector;
Vector3 g_y_vector;
Vector3 g_z_vector;
#endif

// use Catmull-Rom spline to compute a position based on given time t.
// @param control_matrix    A matrix contains control point data in rows.
Vector3 compute_curve_position(double t, const Matrix4& control_matrix)
{
    static const double s=0.5;
    static const Matrix4 catmull_rom_mat( -s  , 2-s,   s-2,  s  ,
                                           2*s, s-3, 3-2*s, -s  ,
                                          -s  , 0.0,   s  ,  0.0, 
                                           0.0, 1.0,   0.0,  0.0 );
    Vector4 u;
    u[3] = 1.0;
    u[2] = u[3] * t;
    u[1] = u[2] * t;
    u[0] = u[1] * t;
 
    Matrix4 mat = catmull_rom_mat * control_matrix;
    Vector3 U( u[0] * mat._m[0][0] + u[1] * mat._m[0][1] + u[2] * mat._m[0][2] + u[3] * mat._m[0][3],
               u[0] * mat._m[1][0] + u[1] * mat._m[1][1] + u[2] * mat._m[1][2] + u[3] * mat._m[1][3],
               u[0] * mat._m[2][0] + u[1] * mat._m[2][1] + u[2] * mat._m[2][2] + u[3] * mat._m[2][3] ) ;
    return U;
}

// This function will calculate proper timestep by moving given distance.
// 
// First this function will find a proper range that contains the timestep.
// Then use a binary search to approximate the timestep.
double KeyFrameCurve::get_timestep_by_distance( const Vector3& cur_pos, 
                                            const double ideal_distance, 
                                            const Matrix4& control_matrix )
{
    int index2 = CP2Index();
    // tentative time step
    double time_step = 0.1;
    double next_t = m_t + time_step;
    Vector3 next_pos = compute_curve_position(next_t - index2, control_matrix);
    double diff = distance(next_pos, cur_pos) - ideal_distance;

    // make sure time step is bigger than needed 
    while (diff < 0) {
        time_step *= 2;
        next_t = m_t + time_step;
        next_pos = compute_curve_position(next_t - index2, control_matrix);
        diff = distance(next_pos, cur_pos) - ideal_distance;
    }

    // binary search in time dimension 
    while (fabs(diff) > 0.001 )
    {
        time_step /= 2.0;
        if (diff > 0) {
            next_t -= time_step;
        } else {
            next_t += time_step;
        }

        next_pos = compute_curve_position(next_t - index2, control_matrix);
        diff = distance(next_pos, cur_pos) - ideal_distance;
        if (time_step < 0.000001) {
            m_t = 1.0;
            break;
        }
    }
    return next_t;
}

// Compute next position for camera
// @param delta_time    elapsed time (real time in program)
// @param cur_pos[in]   current camera position
// @param next_pos[out] next camera position
// @return              next timestep for spline
double KeyFrameCurve::compute_new_position(
    double delta_time, const Vector3& cur_pos,
    Vector3& next_pos)
{
    int index1 = CP1Index();
    int index2 = CP2Index();
    int index3 = CP3Index();

    // compute position
    double speed = (m_t - index2) * m_keyFrameInfos[index2].camera_speed + 
                   (index3 - m_t) * m_keyFrameInfos[index3].camera_speed;
    double ideal_distance = delta_time * speed;

    //Matrix4 control_matrix = build_control_matrix(CP1Index());
    Vector3 cp1 = m_keyFrameInfos[index1  ].camera.position;
    Vector3 cp2 = m_keyFrameInfos[index1+1].camera.position;
    Vector3 cp3 = m_keyFrameInfos[index1+2].camera.position;
    Vector3 cp4 = m_keyFrameInfos[index1+3].camera.position;
    Matrix4 control_matrix( cp1[0], cp1[1], cp1[2], 0.0,
                            cp2[0], cp2[1], cp2[2], 0.0,
                            cp3[0], cp3[1], cp3[2], 0.0,
                            cp4[0], cp4[1], cp4[2], 0.0 );

    double next_t = get_timestep_by_distance(cur_pos, ideal_distance, control_matrix);
    next_pos = compute_curve_position(next_t - index2, control_matrix);
    return next_t;
}

// compute orientation for next position
// @param next_t    next timestep
// @param camera    camera object 
// @param next_x[out]   X Axis of orientation.
// @param next_y[out]   Y Axis of orientation.
// @param next_z[out]   Z Axis of orientation.
void KeyFrameCurve::compute_new_orientation(
    const double next_t, 
    //const Camera& camera,
    Vector3& next_x, Vector3 &next_y, Vector3& next_z
    )
{
    int index1 = CP1Index();
    int index2 = CP2Index();
    Vector3 y1 = m_keyFrameInfos[index1  ].camera.get_up();
    Vector3 y2 = m_keyFrameInfos[index1+1].camera.get_up();
    Vector3 y3 = m_keyFrameInfos[index1+2].camera.get_up();
    Vector3 y4 = m_keyFrameInfos[index1+3].camera.get_up();
    Matrix4 y_mat( y1[0], y1[1], y1[2], 0.0,
                   y2[0], y2[1], y2[2], 0.0,
                   y3[0], y3[1], y3[2], 0.0,
                   y4[0], y4[1], y4[2], 0.0 );

    Vector3 z1 = m_keyFrameInfos[index1  ].camera.get_direction();
    Vector3 z2 = m_keyFrameInfos[index1+1].camera.get_direction();
    Vector3 z3 = m_keyFrameInfos[index1+2].camera.get_direction();
    Vector3 z4 = m_keyFrameInfos[index1+3].camera.get_direction();
    // negative zX since direction is negative z axis
    Matrix4 z_mat( -z1[0], -z1[1], -z1[2], 0.0,
                   -z2[0], -z2[1], -z2[2], 0.0,
                   -z3[0], -z3[1], -z3[2], 0.0,
                   -z4[0], -z4[1], -z4[2], 0.0 );

    next_y = compute_curve_position((next_t - index2), y_mat);
    next_z = compute_curve_position((next_t - index2), z_mat );
    next_x = cross(next_y, next_z);
    next_z = cross(next_x, next_y);
}


// update camera data with Camull-Rom spline according to delta_time.
void KeyFrameCurve::update_camera(Camera& camera, double delta_time)
{
    //camera.orientation = Quaternion::Identity;
    // if run out of curve, start from beginning
    if ( m_keyFrameInfos.size() <= (size_t)CP4Index() ) {
        m_t = 1.0;
        camera.position = InitialCameraPosition();
    }

    // compute next step position and tagent, preparing for double reflection
    Vector3 cur_pos = camera.get_position();
    Vector3 next_pos;
    double next_t = compute_new_position( delta_time, cur_pos, next_pos);

    // compute new orientation
    Vector3 next_x, next_y, next_z;
    compute_new_orientation( next_t, next_x, next_y, next_z );

    // update camera
    m_t = next_t;
    camera.position = next_pos;
    camera.orientation.from_axis(next_x, next_y, next_z);

    // global debugging info
#ifdef  DEBUG_KEY_FRAME_CURVE
    g_x_vector = next_x;
    g_y_vector = next_y;
    g_z_vector = next_z;
#endif
}

void KeyFrameCurve::display_keyframe_info()
{
    dump_keyframe_info(std::cout);
}

void KeyFrameCurve::dump_keyframe_info(std::ostream& os)
{
    size_t num_key_frame = m_keyFrameInfos.size();
    for (size_t i=0; i<num_key_frame; i++) {
        KeyFrameInfo& info = m_keyFrameInfos[i];
        os << info << std::endl;
    }
}

void KeyFrameCurve::save_keyframe_info()
{
    std::ofstream ofs("keyframeinfo.txt", std::ios_base::out);
    dump_keyframe_info(ofs);
    ofs.close();
}

void KeyFrameCurve::load_keyframe_info()
{
    std::ifstream ifs("keyframeinfo.txt", std::ios_base::in);
    KeyFrameInfo info;
    ifs >> info;
    while (!ifs.eof()) {
        m_keyFrameInfos.push_back(info);
        ifs >> info;
    }
    ifs.close();
}

std::ostream& operator<<(std::ostream& os, KeyFrameInfo& info)
{
    Camera& camera = info.camera;
    os << camera.position << " " << camera.orientation << " " << info.camera_speed;
    return os;
}

std::istream& operator>>(std::istream& is, KeyFrameInfo& info)
{
    Vector3& position  = info.camera.position;
    Quaternion& orient = info.camera.orientation;
    char line[256];
    is.getline(line, 256);
    sscanf(line, "(%lf,%lf,%lf) Quaternion(%lf, %lf, %lf, %lf) %lf", 
           &position.x, &position.y, &position.z,
           &orient.w, &orient.x, &orient.y, &orient.z, &info.camera_speed);
    return is;
}

}

