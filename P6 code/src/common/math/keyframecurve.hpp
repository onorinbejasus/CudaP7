#pragma once

#ifndef KEY_FRAME_CURVE_H
#define KEY_FRAME_CURVE_H

#include "common/math/matrix.hpp"
#include "common/math/vector.hpp"
#include "common/math/camera.hpp"
#include "common/math/camera_roam.hpp"
#include <iostream>
#include <fstream>
#include <vector>

namespace aed {

// Key Frame Info structure.
struct KeyFrameInfo
{
    Camera camera;          // Camera position and orientation
    double camera_speed;    // Camera speed at this key frame.
};

/**
 * Key Frame Info structure IO operations. Used to save to/load from disk file.
 */
std::ostream& operator<<(std::ostream& os, KeyFrameInfo& info);
std::istream& operator>>(std::istream& is, KeyFrameInfo& info);


/**
 * This class records the key frame information and time step for current
 * position.
 *
 * There are two ways to get key frame information, one is push runtime
 * information to vector by calling push_keyframe_info(Camera&); another is
 * loading file from disk by invoking load_keyframe_info().
 */
class KeyFrameCurve 
{
private:
    // key frame information. The first and last one are only used to get
    // derivative at end points.
    std::vector<KeyFrameInfo> m_keyFrameInfos;

    // time step for spline. Here use m_t is from 1 to number_keyframe-1 
    // instead of normally [0, 1].
    // index1 = (int)m_t -1 is the first control point of current spline, while
    // index4 = index+3 is the last control point of current spline.
    double m_t; 


public:
    KeyFrameCurve() : m_t(1.0) {}
    ~KeyFrameCurve() {}

    // return the 2nd position as initial position
    Vector3 InitialCameraPosition()
    {
        if (IsReady())
            return m_keyFrameInfos[1].camera.position;
        return Vector3(0, 0, 0);
    }

    // Check if this class is ready to work. Need at least 4 key frames to work.
    bool IsReady() { return m_keyFrameInfos.size() > 4;  }

    // push a keyframe info data
    void push_keyframe_info(Camera& camera)
    {
        KeyFrameInfo info;
        info.camera = camera;
        info.camera_speed  = 2.0;
        m_keyFrameInfos.push_back(info);
    }

    void dump_keyframe_info(std::ostream& os); // dump keyframe info to given stream
    void display_keyframe_info(); // display keyframe information on screen
    void save_keyframe_info();    // save keyframe info to a file
    void load_keyframe_info();    // load keyframe info from a file

    // update camera data with Camull-Rom spline according to delta_time.
    void update_camera(Camera& camera, double delta_time);
    
private:
    int CP1Index() { return (int)m_t-1; }   // current 1st control point index
    int CP2Index() { return (int)m_t  ; }   // current 2nd control point index
    int CP3Index() { return (int)m_t+1; }   // current 3rd control point index
    int CP4Index() { return (int)m_t+2; }   // current 4th control point index

    // move forward by given distance, return the new time step
    double get_timestep_by_distance(
        const Vector3& cur_pos,
        const double ideal_distance,
        const Matrix4& control_matrix);

    // Compute next position for camera according to elapsed delta_time.
    // @param delta_time    elapsed time (real time in program)
    // @param cur_pos[in]   current camera position
    // @param next_pos[out] next camera position
    // @return              next timestep for spline
    double compute_new_position(
        const double delta_time, const Vector3& cur_pos, Vector3& next_pos);

    // compute orientation for next position
    // @param next_t    next timestep
    // @param camera    camera object 
    // @param next_x[out]   X Axis of orientation.
    // @param next_y[out]   Y Axis of orientation.
    // @param next_z[out]   Z Axis of orientation.
    void compute_new_orientation(
            const double next_t,
            Vector3& next_x, Vector3 &next_y, Vector3& next_z );    
};

// XXX: some global variables used to debug. Should remove later.
#ifdef  DEBUG_KEY_FRAME_CURVE
extern Vector3 g_x_vector;
extern Vector3 g_y_vector;
extern Vector3 g_z_vector;
#endif

}

#endif

