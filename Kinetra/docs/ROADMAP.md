# Status and roadmap

## Done (v0.2)
Unified app, single + multi camera, video input, project/take system, raw recording + offline zero-phase reprocessing,
BVH / USD / CSV / UDP / OSC, locked bone lengths, foot lock with leg IK, twist, face (52 ARKit shapes, muscles, markers,
photo profile, .glb characters), **hands and 15 finger joints per hand, 52-joint skeleton, detailed procedural body/hand/head
model, anatomical joint limits, AI add-on (finger-state HMM prior with occlusion handling, body completion network, trainer
with BVH import), guided face scan -> personal face rig -> features-only 3D face, USD point cache**, 8 languages, 46 tests.

## Characters
Rigged humanoids (.glb / .vrm / .gltf) are imported, bone-mapped (Mixamo, Unity/RPM, Unreal, VRM tables), retargeted by
rest-pose alignment (T/A-pose independent), skinned (linear blend skinning), textured and driven by ARKit morph targets. Tested on a
Mixamo character (49 of 52 joints mapped; eyes and jaw need a rig that has them). Not done: dual-quaternion skinning, cloth/hair
physics, corrective shapes, facial bone rigs.

## Honest limits
* The AI is trained on procedural, anatomically limited motion plus whatever you add (your takes, BVH libraries). It will not
  be "perfect": quality grows with real data. Check any dataset licence before shipping a model trained on it (CMU mocap is free
  for use in products but has no finger data; AMASS-style sets are research-only as far as known; SFU is research-only).
* No real camera, real MediaPipe models or GPU were available while building: pipeline, solvers, AI and exports are verified with
  synthetic cameras/landmarks; the OpenGL views and the scan dialog need a check on real hardware.
* The body model is a rigidly skinned procedural mesh, not a sculpted, blend-skinned character (use the 52-joint BVH/USD on yours).
* Not done: hardware time sync, retargeting editor, curve/cleanup editor, Unreal Live Link / Blender add-on, installer and
  licensing, accuracy benchmark against a marker system, per-finger visibility from a dedicated finger model.
