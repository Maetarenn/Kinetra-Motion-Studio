# Architecture

```
CameraSource x N (threads, newest-frame-wins)         video file / camera
        |
CaptureWorker (QThread)  -- one frame set in flight, always the newest --
   ThreadPool:  PoseLandmarker per camera  +  FaceLandmarker (face camera)
        |
   3D body:  single  -> MediaPipe world landmarks (+ pinhole root estimate)
             multi   -> Rig.triangulate (weighted DLT, reprojection outlier rejection)
        |
   BodySolver: One-Euro -> bone lengths locked -> ground -> hinge limits -> foot lock + two-bone IK
               -> quaternions with twist (hips, chest, head, arms, legs, feet)
   FaceTracker: identity matching, One-Euro, neutral / range-of-motion calibration, markers, 25 muscles
        |
   result dict -> UI (frame_ready)   TakeRecorder (raw.npz)   Streamer (UDP JSON / OSC)   video writer
```

* `core/` has no Qt and no MediaPipe: everything there is unit-tested with synthetic data.
* `capture/pipeline.py` accepts injected `models` and sources, so the whole chain runs in tests without hardware.
* World frame of a calibrated rig: metres, y up, floor at y = 0, defined by the floor board.
* Joint convention: +x = subject's left, +z = facing direction, y up. Rest skeleton: arms and legs hang down.
* Offline reprocessing (`core/project.reprocess_take`) uses the same solver on forward-backward filtered raw data.

## v0.2 additions
```
MediaPipe HandLandmarker (per camera) -> assign to body wrists -> triangulate (multi) / world landmarks (single)
   -> HandSolver (palm frame, 15 finger joints) -> HandPoseFilter (GMM + HMM prior)  <- finger_confidence()
BodySolver (13 joints) + hands + face blendshapes -> fullbody.assemble (52 joints) -> anatomy.apply_limits
   -> forward_kinematics -> BodyModel (detailed mesh) / BVH / USD / OSC / UDP
LiveBodyPrior (masked denoising net) fills joints the camera cannot see, before the solver
FaceEnrollment (guided scan) -> FaceRig (identity + personal expression deltas) -> solve() each frame
   -> face_personal.csv + face_points.usda (feature point cache)
```
* AI models live in `kinetra/ai`. Shipped weights: `ai/pretrained`; models trained by the user: `~/.kinetra/ai` (preferred).
* Everything under `core/` and `ai/` is Qt-free and tested with synthetic data.
