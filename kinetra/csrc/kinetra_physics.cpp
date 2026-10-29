#define KINETRA_EXPORTS
#include "kinetra_physics.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {

template <typename T>
inline T clamp_val(T v, T lo, T hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

    float length_sq() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(length_sq()); }

    Vec3 normalized() const {
        float l = length();
        return (l > 1e-7f) ? Vec3(x / l, y / l, z / l) : Vec3(0, 0, 0);
    }
};

struct JointPhysState {
    Vec3 pos;
    Vec3 vel;
    float mass = 1.0f;
    float stiffness = 300.0f;
    float damping = 35.0f;
};

class PhysicsSolverImpl {
public:
    explicit PhysicsSolverImpl(int num_joints, const KinetraPhysicsConfig* config)
        : joints_(num_joints)
    {
        if (config) {
            cfg_ = *config;
        } else {
            cfg_.gravity_y = -9.81f;
            cfg_.ground_y = 0.0f;
            cfg_.floor_restitution = 0.0f;
            cfg_.floor_friction = 0.85f;
            cfg_.default_stiffness = 320.0f;
            cfg_.default_damping = 38.0f;
        }

        for (auto& j : joints_) {
            j.stiffness = cfg_.default_stiffness;
            j.damping = cfg_.default_damping;
        }
    }

    void reset(const float* initial_positions) {
        if (!initial_positions) return;
        for (size_t i = 0; i < joints_.size(); ++i) {
            joints_[i].pos = Vec3(initial_positions[i * 3 + 0],
                                  initial_positions[i * 3 + 1],
                                  initial_positions[i * 3 + 2]);
            joints_[i].vel = Vec3(0.0f, 0.0f, 0.0f);
        }
    }

    void set_ground(float ground_y, float restitution, float friction) {
        cfg_.ground_y = ground_y;
        cfg_.floor_restitution = clamp_val(restitution, 0.0f, 1.0f);
        cfg_.floor_friction = clamp_val(friction, 0.0f, 1.0f);
    }

    void step(float dt, const float* in_targets, float* out_positions, float* out_velocities) {
        if (!in_targets || !out_positions) return;
        if (dt <= 0.0f) dt = 1.0f / 60.0f;
        if (dt > 0.05f) dt = 0.05f; // Clamp large time steps to preserve stability

        for (size_t i = 0; i < joints_.size(); ++i) {
            Vec3 target(in_targets[i * 3 + 0], in_targets[i * 3 + 1], in_targets[i * 3 + 2]);
            auto& j = joints_[i];

            // Spring force towards vision landmark target
            Vec3 err = target - j.pos;
            Vec3 f_spring = err * j.stiffness;

            // Damping force
            Vec3 f_damper = j.vel * (-j.damping);

            // Total acceleration (f = ma)
            Vec3 acc = (f_spring + f_damper) * (1.0f / j.mass);

            // Semi-implicit Euler integration
            j.vel += acc * dt;
            j.pos += j.vel * dt;

            // Floor collision non-penetration constraint
            if (j.pos.y < cfg_.ground_y) {
                j.pos.y = cfg_.ground_y;
                if (j.vel.y < 0.0f) {
                    j.vel.y = -j.vel.y * cfg_.floor_restitution;
                    j.vel.x *= (1.0f - cfg_.floor_friction);
                    j.vel.z *= (1.0f - cfg_.floor_friction);
                }
            }

            // Write output
            out_positions[i * 3 + 0] = j.pos.x;
            out_positions[i * 3 + 1] = j.pos.y;
            out_positions[i * 3 + 2] = j.pos.z;

            if (out_velocities) {
                out_velocities[i * 3 + 0] = j.vel.x;
                out_velocities[i * 3 + 1] = j.vel.y;
                out_velocities[i * 3 + 2] = j.vel.z;
            }
        }
    }

private:
    std::vector<JointPhysState> joints_;
    KinetraPhysicsConfig cfg_;
};

} // namespace

extern "C" {

KINETRA_API void* kinetra_physics_create(int num_joints, const KinetraPhysicsConfig* config) {
    if (num_joints <= 0) return nullptr;
    return new PhysicsSolverImpl(num_joints, config);
}

KINETRA_API void kinetra_physics_destroy(void* solver) {
    if (solver) {
        delete static_cast<PhysicsSolverImpl*>(solver);
    }
}

KINETRA_API void kinetra_physics_reset(void* solver, const float* initial_positions) {
    if (solver) {
        static_cast<PhysicsSolverImpl*>(solver)->reset(initial_positions);
    }
}

KINETRA_API void kinetra_physics_set_ground(void* solver, float ground_y, float restitution, float friction) {
    if (solver) {
        static_cast<PhysicsSolverImpl*>(solver)->set_ground(ground_y, restitution, friction);
    }
}

KINETRA_API void kinetra_physics_step(
    void* solver,
    float dt,
    const float* in_targets,
    float* out_positions,
    float* out_velocities
) {
    if (solver) {
        static_cast<PhysicsSolverImpl*>(solver)->step(dt, in_targets, out_positions, out_velocities);
    }
}

KINETRA_API void kinetra_fast_lbs(
    const float* rest_v,
    const float* rest_n,
    const float* joint_matrices,
    const int* joint_nodes,
    const float* weights,
    int num_verts,
    float* out_v,
    float* out_n
) {
    if (!rest_v || !rest_n || !joint_matrices || !joint_nodes || !weights || !out_v || !out_n || num_verts <= 0) {
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_verts; ++i) {
        const float vx = rest_v[i * 3 + 0];
        const float vy = rest_v[i * 3 + 1];
        const float vz = rest_v[i * 3 + 2];

        const float nx = rest_n[i * 3 + 0];
        const float ny = rest_n[i * 3 + 1];
        const float nz = rest_n[i * 3 + 2];

        float def_vx = 0.0f, def_vy = 0.0f, def_vz = 0.0f;
        float def_nx = 0.0f, def_ny = 0.0f, def_nz = 0.0f;

        for (int k = 0; k < 4; ++k) {
            const int nid = joint_nodes[i * 4 + k];
            const float w = weights[i * 4 + k];
            if (nid < 0 || w <= 0.0f) continue;

            const float* M = &joint_matrices[nid * 16];

            // Transform point: (M * [vx, vy, vz, 1])
            const float px = M[0] * vx + M[1] * vy + M[2] * vz + M[3];
            const float py = M[4] * vx + M[5] * vy + M[6] * vz + M[7];
            const float pz = M[8] * vx + M[9] * vy + M[10] * vz + M[11];

            // Transform normal: (M * [nx, ny, nz, 0])
            const float tnx = M[0] * nx + M[1] * ny + M[2] * nz;
            const float tny = M[4] * nx + M[5] * ny + M[6] * nz;
            const float tnz = M[8] * nx + M[9] * ny + M[10] * nz;

            def_vx += w * px;
            def_vy += w * py;
            def_vz += w * pz;

            def_nx += w * tnx;
            def_ny += w * tny;
            def_nz += w * tnz;
        }

        // Normalize normal
        const float len = std::sqrt(def_nx * def_nx + def_ny * def_ny + def_nz * def_nz);
        if (len > 1e-7f) {
            def_nx /= len;
            def_ny /= len;
            def_nz /= len;
        }

        out_v[i * 3 + 0] = def_vx;
        out_v[i * 3 + 1] = def_vy;
        out_v[i * 3 + 2] = def_vz;

        out_n[i * 3 + 0] = def_nx;
        out_n[i * 3 + 1] = def_ny;
        out_n[i * 3 + 2] = def_nz;
    }
}

} // extern "C"
