/**
 * @file camera_roam.cpp
 * @brief CameraRoamControl class
 *
 * @author Zeyang Li (zeyangl)
 * @author Eric Butler (edbutler)
 * @description   this file implements a free roam camera
 */

#include "camera_roam.hpp"
#include "common/math/matrix.hpp"
#include <algorithm>

namespace aed {

static const real_t DirectionTable[] = { 0.0, 1.0, -1.0 };
static const int anchor[2]={ 256, 256 };
//HACK: some angle restriction for hiding incomplete texture, 
//      reset constant data after someone messing it up 
static const Vector3 default_position( 0, 20, 40 );
static const real_t default_pitch_angle = -PI / 7.0;
real_t mouse_x, mouse_y;

CameraRoamControl::CameraRoamControl()
{
    roam_controller = NULL;
    reset_camera(); 
//    camera.rotate(Vector3::UnitZ, PI/10.0);
}

CameraRoamControl::~CameraRoamControl()
{
    ClearRoamController();
}

void CameraRoamControl::set_dir( bool pressed, int index, Direction newdir )
{
    // fi pressed set, otherwise only undo it if other direction is not the one being pressed
    if ( pressed )
        direction[index] = newdir;
    else if ( direction[index] == newdir )
        direction[index] = DZERO;
}

void CameraRoamControl::toggle_roam_mode() {
}
void CameraRoamControl::reset_camera() {

    camera.position = default_position;
    camera.orientation = Quaternion::Identity;
    camera.pitch( default_pitch_angle );

    direction[0] = DZERO;
    direction[1] = DZERO;
    direction[2] = DZERO;
    rotation = RNONE;

    ClearRoamController();
    roam_controller = new HPVTRoam(camera);
    roam_mode = HPVT;
}

void CameraRoamControl::handle_event( const SDL_Event& event )
{
    int newidx = -1;
    Direction newdir = DZERO;

    switch ( event.type )
    {
    case SDL_KEYDOWN:
        if (event.key.keysym.sym == SDLK_o) {
            reset_camera();
            break;
        }
        if (event.key.keysym.sym == SDLK_p) {
            toggle_roam_mode();
            break;
        }
    case SDL_KEYUP:
        switch( event.key.keysym.sym )
        {
            case SDLK_w:
                newidx = 2;
                newdir = DNEG;
                break;
            case SDLK_s:
                newidx = 2;
                newdir = DPOS;
                break;
            case SDLK_a:
                newidx = 0;
                newdir = DNEG;
                break;
            case SDLK_d:
                newidx = 0;
                newdir = DPOS;
                break;
            case SDLK_q:
                newidx = 1;
                newdir = DNEG;
                break;
            case SDLK_e:
                newidx = 1;
                newdir = DPOS;
                break;
            case SDLK_UP:
                roam_controller->changeTranslationSpeed( 2 );
                break;
            case SDLK_DOWN:
                roam_controller->changeTranslationSpeed( 0.5 );
                break;
            default:
                newidx = -1;
                break;
        }

        if ( newidx != -1 ) {
            set_dir( event.key.state == SDL_PRESSED, newidx, newdir );
        }
        break;

    case SDL_MOUSEBUTTONDOWN:
        // enable rotation
        if ( event.button.button == SDL_BUTTON_LEFT )
            rotation = RPITCHYAW;
        else if ( event.button.button == SDL_BUTTON_MIDDLE )
            rotation = RROLL;
        break;

    case SDL_MOUSEBUTTONUP:
        // disable rotation
        if ( event.button.button == SDL_BUTTON_LEFT && rotation == RPITCHYAW )
            rotation = RNONE;
        else if ( event.button.button == SDL_BUTTON_MIDDLE && rotation == RROLL )
            rotation = RNONE;
        break;

    case SDL_MOUSEMOTION:
#if 0
        if(roam_mode == RPY) {
            if ( rotation == RPITCHYAW ) {
                camera.yaw( -RotationSpeed * event.motion.xrel );
                camera.pitch( -RotationSpeed * event.motion.yrel );
            } else if ( rotation == RROLL ) {
                camera.roll( RotationSpeed * event.motion.yrel );
            }
            //SDL_ShowCursor( SDL_ENABLE );
        }
        else {//if(roam_mode == FPS){ // FPS mode
            mouse_x = 0.003 * (real_t)( event.motion.x - anchor[0] );
            mouse_y = 0.0025 * (real_t)( event.motion.y - anchor[1] );
            /*if(roam_mode == FPS){
              camera.rotate( Vector3::UnitY, -RotationSpeed * mouse_x );
              }
              camera.pitch( -RotationSpeed * mouse_y );*/

            //SDL_WarpMouse( anchor[0], anchor[1] );
            //SDL_ShowCursor( SDL_DISABLE );
        }
#endif
        break;

    default:
        break;
    }
}

void RPYRoam::update(Camera& camera, real_t dt, real_t disp1, real_t disp2, real_t disp3)
{
    real_t dist = TranslationSpeed * dt;
    // update the position based on keys
    // no need to update based on mouse, that's done in the event handling
    Vector3 displacement( disp1, disp2, disp3 );
    camera.translate( displacement * dist );

    //if( roam_mode == FPS ) {
    //    camera.rotate( Vector3::UnitY, -RotationSpeed * dt * mouse_x );
    //    camera.pitch( -RotationSpeed * dt * mouse_y );
    //}

    // HACK: use phi angle in polar coordinate system to limit the camera movement, in case the camera
    //       roam into the place where texture are incomplete
    Vector3 cam_pos = camera.get_position();
    real_t current_r = length( cam_pos - Vector3::Zero );

    real_t current_phi = acos( (cam_pos.y - 0.0) / current_r );
    if( current_phi <= min_phi ) {
        camera.position.y = current_r * cos( min_phi );
    }
    else if( current_phi >= max_phi ) {
        camera.position.y = current_r * cos( max_phi );
    }

    if( current_r < closest_r ) {
        camera.position = closest_r / current_r * ( cam_pos - Vector3::Zero );
    }
}

void HPVTRoam::update(Camera& camera, real_t dt, real_t delta_theta, real_t delta_phi, real_t delta_r)
{
    static real_t old_r = 0, old_t = 0, old_p = 0;
    real_t acc = 0.002 * dt;

    real_t delta_theta_n = delta_theta;
    if ( fabs( delta_theta - old_t ) > acc && acc > 0.00 ) {
        real_t dir = fabs( delta_theta - old_t ) / ( delta_theta - old_t );
        delta_theta_n = old_t + dir * acc;
        delta_theta = delta_theta == 0 ? -dir : dir;
    }
    old_t = delta_theta_n;

    real_t delta_r_n = delta_r;
    if ( fabs( delta_r - old_r ) > acc && acc > 0.00 ) {
        real_t dir = fabs( delta_r - old_r ) / ( delta_r - old_r );
        delta_r_n = old_r + dir * acc;
        delta_r = delta_r == 0 ? -dir : dir;
    }
    old_r = delta_r_n;

    real_t delta_phi_n = delta_phi;
    if ( fabs( delta_phi - old_p ) > acc && acc > 0.00 ) {
        real_t dir = fabs( delta_phi - old_p ) / ( delta_phi - old_p );
        delta_phi_n = old_p + dir * acc;
        delta_phi = delta_phi == 0 ? -dir : dir;
    }
    old_p = delta_phi_n;

    real_t dist_theta = TranslationSpeed * dt * fabs( delta_theta_n );
    real_t dist_phi = TranslationSpeed * dt * fabs( delta_phi_n ) * 0.5;
    real_t dist_r = TranslationSpeed * dt * fabs( delta_r_n ) ;

    // compute camera posiiton after movement
    // --- pivot radius
    pivot_r += delta_r * dist_r;        
    if( pivot_r < closest_r ) 
        pivot_r = closest_r;

    // --- pivot theta
    pivot_theta -= delta_theta * dist_theta * PIV_RotationSpeed;

    // --- pivot phi
    real_t tentative_pivot_phi = pivot_phi - delta_phi * dist_phi * PIV_RotationSpeed;
    real_t next_pivot_phi = std::min(std::max(tentative_pivot_phi, min_phi), max_phi);
    real_t phi_offset = pivot_phi - next_pivot_phi;
    pivot_phi = next_pivot_phi;
    // --- camera position
    camera.position = Vector3( pivot_r * cos( pivot_theta ) * sin( pivot_phi ), 
            pivot_r * cos( pivot_phi ), 
            pivot_r * sin( pivot_theta ) * sin( pivot_phi ) );
    // rotate camera
    Vector3 old_cam_dir = camera.get_direction();
    Vector3 new_cam_dir = normalize( Vector3( 0, 0, 0 ) - camera.position );
    real_t theta_offset = acos( dot( normalize(Vector3(old_cam_dir.x, 0, old_cam_dir.z)), 
                normalize(Vector3(new_cam_dir.x, 0, new_cam_dir.z)) ) );

    // XXX: I doesn't work well if I directly call 
    //    camera.rotate( Vector3::UnitY, delat_theta * theta_offset );
    // The backend crashes in this situation. Weird.
    if(delta_theta == 1) {
        camera.rotate( Vector3::UnitY,  theta_offset );
    } else if(delta_theta == -1){
        camera.rotate( Vector3::UnitY, -theta_offset );
    }

    camera.pitch( -phi_offset );
    camera.pitch( -RotationSpeed * dt * mouse_y );
}

void VPVTRoam::update(Camera&, real_t, real_t, real_t, real_t)
{
/*
	Vector3 new_pos = trans_mat * camera.position;
	double  alpha = atan2(new_pos.x, -new_pos.z);
	if (alpha < PI/30.0)
	{
		Vector3 next_pos = new_pos + Vector3::UnitX * dt * TranslationSpeed;
		double next_alpha = atan2(next_pos.x, -next_pos.z);
		if (next_alpha < PI/30.0) {
			camera.position = normalized(next_pos) * pivot_r;
		} else {
			new_pos.x = pivot_r * sin(PI/30);
			new_pos.z = 
		}
	}
	double x = camera.position.x;
	double y = camera.position.y;
    double z = camera.position.z;

    double angle = atan2(y, x);
    double delta_angle = RotationSpeed * dt * delta_theta;
    double radius = sqrt(x*x + y*y);
    double xx = radius * cos(angle + delta_angle);
    double yy = radius * sin(angle + delta_angle);

    camera.position.x = xx;
    camera.position.y = yy;
    printf("original  x = %f, y = %f.\n", x, y);
    printf("change to x = %f, y = %f.\n", camera.position.x, camera.position.y);
    
    Vector3 dir = normalize( camera.position );
    Vector3 up(0, 1, 0);
    Vector3 right = normalize(cross(up, dir));
    up = normalize(cross(dir, right));
    Matrix4 mat4;
    for (int i=0; i<3; i++)
        mat4._m[i][0] = right[i];
    for (int i=0; i<3; i++)
        mat4._m[i][1] = up[i];
    for (int i=0; i<3; i++)
        mat4._m[i][2] = dir[i];
    for (int i=0; i<3; i++)
        mat4._m[3][i] = 0;
    for (int i=0; i<3; i++)
        mat4._m[i][3] = 0;
    mat4._m[3][3] = 1;
    camera.orientation = Quaternion(mat4);
    */
}


void CameraRoamControl::update( real_t dt )
{
    // RPY and FPS share update function
    roam_controller->update(
            camera,
            dt, 
            DirectionTable[direction[0]],
            DirectionTable[direction[1]],
            DirectionTable[direction[2]]
    );
}

void CameraRoamControl::get_camera( Camera& camera ) {
	camera = this->camera;
}

} /* aed */
