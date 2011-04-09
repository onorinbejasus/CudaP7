/**
 * @file camera_roam.cpp
 * @brief CameraRoamControl class
 *
 * @author Zeyang Li (zeyangl)
 */

#ifndef _CAMERAROAM_HPP_
#define _CAMERAROAM_HPP_

#include "common/math/camera.hpp"
#include "common/math/matrix.hpp"
#include <SDL/SDL.h>

namespace aed {
class RoamController
{
public:
    // XXX changed max phi so we can move all around the motorcycle. Originally 7.0/18.0*PI
    RoamController() : min_phi(0.2), max_phi(PI), closest_r(20), TranslationSpeed(0.02), RotationSpeed(0.003), curr_trans_speed(0), curr_rot_speed(0) {}
    virtual ~RoamController() {}

    virtual void update(Camera& camera, double dt, real_t, real_t, real_t) = 0;

    virtual void reset(Camera&) {
        TranslationSpeed = 0.02;
        RotationSpeed = 0.003;
    }
    void changeTranslationSpeed(double factor) { TranslationSpeed *= factor; }

protected:
    const real_t min_phi;
    const real_t max_phi;
    const real_t closest_r;
    real_t TranslationSpeed;
    real_t RotationSpeed;
    real_t curr_trans_speed;
    real_t curr_rot_speed;
};

class RPYRoam : public RoamController
{
public:
    RPYRoam() {}
    void update(Camera& camera, double dt, real_t disp1, real_t disp2, real_t disp3);
};

class HPVTRoam : public RoamController
{
public:
    HPVTRoam(Camera& camera) : PIV_RotationSpeed(0.05)
    { 
        attach(camera); 
    }

    void update(Camera& camera, double dt, real_t delta_theta, real_t delta_phi, real_t delta_r);
    virtual void reset(Camera& camera)
    {
        RoamController::reset(camera);
        attach(camera);
    }

    void attach(Camera& camera) {
        Vector3 cam_pos = camera.get_position();
        pivot_r = length( cam_pos );
        pivot_theta = atan2( cam_pos.z - 0.0, cam_pos.x - 0.0);
        if(pivot_theta < 0.0) {
            pivot_theta += 2*PI;
        }
        pivot_phi = acos( (cam_pos.y - 0.0) / pivot_r );
    }

private:
    const real_t PIV_RotationSpeed;
    real_t pivot_r, pivot_theta, pivot_phi;
};

class VPVTRoam : public RoamController
{
public:
    VPVTRoam(Camera& camera) : PIV_RotationSpeed(0.05)
    { 
        attach(camera); 
    }

    void update(Camera& camera, double dt, real_t delta_theta, real_t delta_phi, real_t delta_r);
    virtual void reset(Camera& camera)
    {
        RoamController::reset(camera);
        attach(camera);
    }

    void attach(Camera& camera) {
        Vector3 cam_pos = camera.get_position();
        pivot_r = length(cam_pos);
        
        Vector3 z_axis = - cam_pos;
        Vector3 y_axis(0, 1, 0);
        Vector3 x_axis = cross(y_axis, z_axis);
        y_axis = cross(z_axis, x_axis);
        
	    for (int i=0; i<3; i++)
    	    trans_mat._m[i][0] = x_axis[i];
	    for (int i=0; i<3; i++)
    	    trans_mat._m[i][1] = y_axis[i];
	    for (int i=0; i<3; i++)
    	    trans_mat._m[i][2] = z_axis[i];
	    for (int i=0; i<3; i++)
	        trans_mat._m[3][i] = 0;
    	for (int i=0; i<3; i++)
        	trans_mat._m[i][3] = 0;
	    trans_mat._m[3][3] = 1;

        pivot_theta = 0;
    }

private:
    const real_t PIV_RotationSpeed;
    real_t  pivot_r, pivot_theta;
    Matrix4 trans_mat;
};


class CameraRoamControl
{
public:

    CameraRoamControl();
    ~CameraRoamControl();

    void update( real_t dt );
    void handle_event( const SDL_Event& event );
    void toggle_roam_mode();
    void reset_camera();
	void get_camera( Camera &camera );
    // the camera of this control
    Camera camera;

private:
    
    enum Direction { DZERO=0, DPOS=1, DNEG=2 };
    enum Rotation { RNONE, RPITCHYAW, RROLL };
    enum Roam_mode { RPY, FPS, HPVT, VPVT };
    void set_dir( bool pressed, int index, Direction newdir );
    void ClearRoamController() {
        if (roam_controller)
        {
            delete roam_controller;
            roam_controller = NULL;
        }
    }

    // current directions in the local camera axes, (x, y, z)
    Direction direction[3];
    // the current rotation
    Rotation rotation;

    Roam_mode roam_mode;
    RoamController* roam_controller;
};

} /* aed */

#endif /* _CAMERAROAM_HPP_ */

