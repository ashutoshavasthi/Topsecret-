/* SkyMind dodge reflex: onboard core. Mirrors skymind/dodge.py; see dodge_reflex.h. */
#include "dodge_reflex.h"

#include <math.h>
#include <string.h>

#define DR_G (-9.81f)
#define DR_SAMPLES 71
#define DR_WINDOW 0.35f
#define DR_STALE 1.0f

static float dot3(const float a[3], const float b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static float norm3(const float a[3]) { return sqrtf(dot3(a, a)); }

void dr_init(dodge_reflex_t *r, float body_radius, float ceiling, float min_altitude) {
    memset(r, 0, sizeof(*r));
    r->body_radius = body_radius;
    r->ceiling = ceiling;
    r->min_altitude = min_altitude;
    r->horizon = 0.7f;
    r->margin = 0.18f;
    r->accel_up = 6.9f;
    r->accel_down = 9.3f;
    r->accel_side = 8.2f;
    r->lag_vertical = 0.06f;
    r->lag_side = 0.14f;
    r->escape_time = 0.45f;
    r->escape_speed = 2.5f;
}

/* Distance covered after t s accelerating at a (after lag) up to vmax. */
static float escape_travel(float a, float vmax, float t, float lag) {
    float tau = t - lag;
    if (tau <= 0.0f || a <= 1e-6f) return 0.0f;
    float t_sat = vmax / a;
    return tau < t_sat ? 0.5f * a * tau * tau : 0.5f * vmax * t_sat + vmax * (tau - t_sat);
}

static int unit3(const float v[3], float out[3]) {
    float n = norm3(v);
    if (n <= 1e-6f) return 0;
    for (int k = 0; k < 3; k++) out[k] = v[k] / n;
    return 1;
}

/* Closest distance to the ball when escaping along unit d; -1 if the escape leaves floor/ceiling. */
static float predicted_miss(const dodge_reflex_t *r, const float d[3], const float traj[][3], const float *ts,
                            float altitude) {
    float dh = sqrtf(d[0] * d[0] + d[1] * d[1]), dz = d[2];
    float a_v = dz > 0.0f ? r->accel_up : r->accel_down;
    float as = r->accel_side > 1e-6f ? r->accel_side : 1e-6f;
    if (a_v < 1e-6f) a_v = 1e-6f;
    float inv = (dh / as) * (dh / as) + (fabsf(dz) / a_v) * (fabsf(dz) / a_v);
    float accel = inv > 0.0f ? 1.0f / sqrtf(inv) : 0.0f;
    float lag = fabsf(dz) > 0.9f ? r->lag_vertical : r->lag_side;
    float final_z = altitude + dz * escape_travel(accel, r->escape_speed, r->escape_time, lag);
    if (final_z < r->min_altitude - 0.05f || final_z > r->ceiling) return -1.0f;
    float best = 1e9f;
    for (int i = 0; i < DR_SAMPLES; i++) {
        float s = escape_travel(accel, r->escape_speed, ts[i], lag);
        float p[3] = {traj[i][0] - s * d[0], traj[i][1] - s * d[1], traj[i][2] - s * d[2]};
        float dist = norm3(p);
        if (dist < best) best = dist;
    }
    return best;
}

/* Ballistic prediction of the tracked object's relative position/velocity at time t. */
static void propagate(const dodge_reflex_t *r, float t, float rel[3], float vel[3]) {
    float dt = t - r->t_last;
    for (int k = 0; k < 3; k++) {
        float g = (k == 2) ? DR_G : 0.0f;
        rel[k] = r->rel[k] + r->vel[k] * dt + 0.5f * g * dt * dt;
        vel[k] = r->vel[k] + g * dt;
    }
}

static void start_track(dodge_reflex_t *r, float t, const float rel[3], float sigma) {
    r->active = 1;
    r->n_hist = 1;
    r->hist[0].t = t;
    memcpy(r->hist[0].rel, rel, sizeof(float) * 3);
    r->hist[0].sigma = sigma;
    r->t_last = t;
    memcpy(r->rel, rel, sizeof(float) * 3);
    r->has_vel = 0;
    r->n_fit = 1;
}

/* Weighted least squares of rel(t) = p + v*(t - t_last) + 0.5*g*(t - t_last)^2 over the window. */
static void fit(dodge_reflex_t *r) {
    float s0 = 0, s1 = 0, s2 = 0, y0[3] = {0}, y1[3] = {0};
    int n = 0;
    for (int i = 0; i < r->n_hist; i++) {
        const dr_meas_t *m = &r->hist[i];
        float dt = m->t - r->t_last;
        if (-dt > DR_WINDOW) continue;
        float sg = m->sigma > 1e-3f ? m->sigma : 1e-3f;
        float w = 1.0f / (sg * sg);
        n++;
        s0 += w;
        s1 += w * dt;
        s2 += w * dt * dt;
        for (int k = 0; k < 3; k++) {
            float y = m->rel[k] - ((k == 2) ? 0.5f * DR_G * dt * dt : 0.0f);
            y0[k] += w * y;
            y1[k] += w * y * dt;
        }
    }
    r->n_fit = n;
    if (n < 3) return;
    float det = (s0 + 1e-9f) * (s2 + 1e-9f) - s1 * s1;
    if (fabsf(det) < 1e-12f) return;
    for (int k = 0; k < 3; k++) {
        r->rel[k] = ((s2 + 1e-9f) * y0[k] - s1 * y1[k]) / det;
        r->vel[k] = ((s0 + 1e-9f) * y1[k] - s1 * y0[k]) / det;
    }
    r->has_vel = 1;
}

void dr_measure(dodge_reflex_t *r, float t, const float rel[3], float sigma) {
    if (!r->active) {
        start_track(r, t, rel, sigma);
        return;
    }
    float predicted[3], vel[3], gate;
    if (r->has_vel) {
        propagate(r, t, predicted, vel);
        gate = 0.6f;
    } else {
        memcpy(predicted, r->hist[r->n_hist - 1].rel, sizeof(float) * 3);
        gate = 1.0f;
    }
    float d[3] = {rel[0] - predicted[0], rel[1] - predicted[1], rel[2] - predicted[2]};
    if (t - r->t_last > DR_STALE || norm3(d) > gate + 2.0f * sigma) {
        start_track(r, t, rel, sigma);  /* a different object, or lost: start over */
        return;
    }
    if (r->n_hist == DR_MAX_HISTORY) {
        memmove(&r->hist[0], &r->hist[1], sizeof(dr_meas_t) * (DR_MAX_HISTORY - 1));
        r->n_hist--;
    }
    dr_meas_t *m = &r->hist[r->n_hist++];
    m->t = t;
    memcpy(m->rel, rel, sizeof(float) * 3);
    m->sigma = sigma;
    r->t_last = t;
    fit(r);
}

int dr_check(dodge_reflex_t *r, float t, float altitude) {
    if (!r->active || !r->has_vel || r->n_fit < 3 || t - r->t_last > DR_STALE) return 0;
    float rel[3], vel[3];
    propagate(r, t, rel, vel);

    /* Closest approach over the horizon (gravity included). */
    float ts[DR_SAMPLES], traj[DR_SAMPLES][3];
    int best = 0;
    float best_d = 1e9f, closest[3] = {0};
    for (int i = 0; i < DR_SAMPLES; i++) {
        ts[i] = r->horizon * (float)i / (float)(DR_SAMPLES - 1);
        traj[i][0] = rel[0] + vel[0] * ts[i];
        traj[i][1] = rel[1] + vel[1] * ts[i];
        traj[i][2] = rel[2] + vel[2] * ts[i] + 0.5f * DR_G * ts[i] * ts[i];
        float d = norm3(traj[i]);
        if (d < best_d) {
            best_d = d;
            best = i;
            memcpy(closest, traj[i], sizeof(closest));
        }
    }
    if (best_d > r->body_radius + r->margin || (best == 0 && dot3(rel, vel) > 0.0f)) return 0;

    float vn = norm3(vel);
    if (vn < 1e-6f) vn = 1e-6f;
    float vdir[3] = {vel[0] / vn, vel[1] / vn, vel[2] / vn};
    float neg[3] = {-closest[0], -closest[1], -closest[2]};
    float along = dot3(neg, vdir);
    float perp[3] = {neg[0] - along * vdir[0], neg[1] - along * vdir[1], neg[2] - along * vdir[2]};

    /* Candidates, in the same order as the Python reference. */
    float cands[7][3];
    int nc = 0;
    cands[nc][0] = 0; cands[nc][1] = 0; cands[nc][2] = -1.0f; nc++;
    cands[nc][0] = 0; cands[nc][1] = 0; cands[nc][2] = 1.0f; nc++;
    float side_raw[3] = {vdir[1], -vdir[0], 0.0f}; /* cross(vdir, z) */
    float side[3];
    if (unit3(side_raw, side)) {
        memcpy(cands[nc++], side, sizeof(side));
        float neg_side[3] = {-side[0], -side[1], -side[2]};
        memcpy(cands[nc++], neg_side, sizeof(neg_side));
    }
    float p[3];
    if (unit3(perp, p)) {
        memcpy(cands[nc++], p, sizeof(p));
        float pd[3] = {p[0], p[1], p[2] - 1.0f}, pu[3] = {p[0], p[1], p[2] + 1.0f}, u[3];
        if (unit3(pd, u)) memcpy(cands[nc++], u, sizeof(u));
        if (unit3(pu, u)) memcpy(cands[nc++], u, sizeof(u));
    }

    int chosen = -1;
    float best_score = -1.0f;
    for (int c = 0; c < nc; c++) {
        float miss = predicted_miss(r, cands[c], (const float(*)[3])traj, ts, altitude);
        if (miss < 0.0f) continue;
        float score = miss + (cands[c][2] < -0.5f ? 0.01f : 0.0f); /* dropping wins ties */
        if (score > best_score) {
            best_score = score;
            chosen = c;
        }
    }
    if (chosen < 0) return 0;
    memcpy(r->escape_dir, cands[chosen], sizeof(r->escape_dir));
    r->triggers++;
    return 1;
}

int dr_tof_zones_to_threat(const unsigned short prev_mm[16], const unsigned short cur_mm[16], float dt,
                           float fov_h, float fov_v, float yaw, float heading, const float own_vel[3],
                           float min_closing, float rel[3], float *sigma) {
    if (dt <= 1e-4f) return 0;
    int best = -1;
    float best_excess = 0.0f, best_dir[3] = {0}, best_r = 0.0f;
    for (int z = 0; z < 16; z++) {
        if (prev_mm[z] == 0 || cur_mm[z] == 0) continue;
        int row = z / 4, col = z % 4;
        float az = heading + fov_h * (0.5f - ((float)col + 0.5f) / 4.0f); /* col 0 = left = +az */
        float el = fov_v * (0.5f - ((float)row + 0.5f) / 4.0f);         /* row 0 = top  = +el */
        float a = yaw + az;
        float dir[3] = {cosf(el) * cosf(a), cosf(el) * sinf(a), sinf(el)};
        float r_prev = prev_mm[z] * 1e-3f, r_cur = cur_mm[z] * 1e-3f;
        float closing = (r_prev - r_cur) / dt;     /* how fast that zone's range shrinks */
        float own = dot3(own_vel, dir);            /* what our own motion explains */
        float excess = closing - own;
        if (excess > min_closing && excess > best_excess) {
            best = z;
            best_excess = excess;
            best_r = r_cur;
            memcpy(best_dir, dir, sizeof(best_dir));
        }
    }
    if (best < 0) return 0;
    for (int k = 0; k < 3; k++) rel[k] = best_r * best_dir[k];
    *sigma = 0.03f + best_r * (fov_h / 4.0f) / 3.464f; /* range noise + zone width / sqrt(12) */
    return 1;
}
