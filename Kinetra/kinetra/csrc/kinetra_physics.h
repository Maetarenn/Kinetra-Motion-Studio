#ifndef KINETRA_PHYSICS_H
#define KINETRA_PHYSICS_H

#ifdef _WIN32
    #ifdef KINETRA_EXPORTS
        #define KINETRA_API __declspec(dllexport)
    #else
        #define KINETRA_API __declspec(dllimport)
    #endif
#else
    #define KINETRA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float gravity_y;
    float ground_y;
    float floor_restitution;
    float floor_friction;
    float default_stiffness;
    float default_damping;
} KinetraPhysicsConfig;

/**
 * Creates a new physical dynamics solver instance.
 */
KINETRA_API void* kinetra_physics_create(int num_joints, const KinetraPhysicsConfig* config);

/**
 * Destroys the solver instance.
 */
KINETRA_API void kinetra_physics_destroy(void* solver);

/**
 * Resets joint positions and zeros out velocities.
 */
KINETRA_API void kinetra_physics_reset(void* solver, const float* initial_positions);

/**
 * Sets floor ground plane height and contact properties.
 */
KINETRA_API void kinetra_physics_set_ground(void* solver, float ground_y, float restitution, float friction);

/**
 * Steps the physics simulation forward by dt (seconds).
 * in_targets: [num_joints * 3] input target positions from vision/mocap
 * out_positions: [num_joints * 3] output simulated physics positions
 * out_velocities: [num_joints * 3] optional output velocities (can be NULL)
 */
KINETRA_API void kinetra_physics_step(
    void* solver,
    float dt,
    const float* in_targets,
    float* out_positions,
    float* out_velocities
);

/**
 * High-performance Linear Blend Skinning (LBS) C++ implementation.
 * rest_v: [num_verts * 3]
 * rest_n: [num_verts * 3]
 * joint_matrices: [num_nodes * 16] 4x4 row-major transform matrices
 * joint_nodes: [num_verts * 4] node indices for 4 bone influences
 * weights: [num_verts * 4] skinning weights
 * out_v: [num_verts * 3] deformed vertices
 * out_n: [num_verts * 3] deformed normals
 */
KINETRA_API void kinetra_fast_lbs(
    const float* rest_v,
    const float* rest_n,
    const float* joint_matrices,
    const int* joint_nodes,
    const float* weights,
    int num_verts,
    float* out_v,
    float* out_n
);

#ifdef __cplusplus
}
#endif

#endif // KINETRA_PHYSICS_H
