/*
 * SkyMind dodge reflex: onboard core (C99, no malloc, float only).
 *
 * The same algorithm as skymind/dodge.py (DodgeReflex), so what the simulator measures
 * is what the drone runs. Feed it threat measurements (relative position of a moving
 * object, world frame, metres) from any sensor: multizone ToF, camera looming, radar.
 * Call dr_check() every control cycle. When it returns 1, fly `escape_dir` at full
 * effort for escape_time seconds, then hold where you end up.
 *
 * The escape direction is chosen model-predictively: each candidate (drop, climb, left,
 * right, away from the ball's line and diagonals) is simulated with the airframe's real
 * acceleration, motor lag and top speed, and the one with the largest predicted miss
 * distance that stays between floor and ceiling wins.
 *
 * tests/test_dodge.py compiles this file with the host C compiler and checks that it
 * makes the same decisions as the Python reference.
 */
#ifndef SKYMIND_DODGE_REFLEX_H
#define SKYMIND_DODGE_REFLEX_H

#ifdef __cplusplus
extern "C" {
#endif

#define DR_MAX_HISTORY 10

typedef struct {
    float t;
    float rel[3];
    float sigma;
} dr_meas_t;

typedef struct {
    /* configuration */
    float body_radius;  /* m, prop tips */
    float ceiling;      /* m, highest allowed altitude */
    float min_altitude; /* m */
    float horizon;      /* s ahead to check for a collision */
    float margin;       /* m extra clearance wanted */
    /* what the airframe can do (see ReflexConfig.from_vehicle in skymind/dodge.py) */
    float accel_up, accel_down, accel_side; /* m/s^2 */
    float lag_vertical, lag_side;           /* s before the motion starts */
    float escape_time, escape_speed;        /* s, m/s */
    /* track */
    int active;
    dr_meas_t hist[DR_MAX_HISTORY];
    int n_hist;
    int n_fit;
    float t_last;
    int has_vel;
    float rel[3];
    float vel[3];
    /* output */
    float escape_dir[3];
    int triggers;
} dodge_reflex_t;

void dr_init(dodge_reflex_t *r, float body_radius, float ceiling, float min_altitude);
void dr_measure(dodge_reflex_t *r, float t, const float rel[3], float sigma);
/* Returns 1 if the tracked object will hit us within the horizon; escape_dir is then set. */
int dr_check(dodge_reflex_t *r, float t, float altitude);

/*
 * Multizone ToF (VL53L5CX/L7CX, 4x4 zones) -> threat measurement.
 * Walls and furniture also return ranges, so a zone only counts as a threat when its
 * range shrinks much faster than our own motion explains (closing speed > min_closing).
 *   prev_mm/cur_mm: 16 zone ranges, 0 = no valid target; row-major, row 0 = top, col 0 = left
 *   dt: s between frames; fov_h/fov_v: rad; yaw: rad (world); heading: sensor yaw offset (rad)
 *   own_vel: our world-frame velocity (m/s)
 * Returns 1 and fills rel (world frame, m) and *sigma for the fastest-closing zone.
 */
int dr_tof_zones_to_threat(const unsigned short prev_mm[16], const unsigned short cur_mm[16], float dt,
                           float fov_h, float fov_v, float yaw, float heading, const float own_vel[3],
                           float min_closing, float rel[3], float *sigma);

#ifdef __cplusplus
}
#endif
#endif
