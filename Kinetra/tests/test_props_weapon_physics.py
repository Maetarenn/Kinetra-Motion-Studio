"""Unit tests for Weapon/Prop Tracker, Physics & Collision, Audio LipSync, and MultiCam Auto-Sync."""
import numpy as np
import pytest

from kinetra.ai.prop_tracker import PropTracker, PropResult
from kinetra.core.physics_props import PhysicsWorld, HitParticle, HitEvent
from kinetra.ai.audio_lipsync import AudioLipSync
from kinetra.capture.autocalib import MultiCamAutoSync
from kinetra.core.constants import BLENDSHAPE_NAMES, NBS
from kinetra.ui.stage import StuntProp


def test_prop_tracker_color_tape_detection():
    tracker = PropTracker(color="green", kind="sword", side="R")

    # Create dummy 480x640 frame
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    # Wrist at (320, 240) in pixels -> normalized (0.5, 0.5)
    wrist_img = np.array([0.5, 0.5, 0.0, 1.0])
    wrist_3d = np.array([0.2, 1.0, 0.0])

    # Paint a bright Neon Green tape patch at (420, 180)
    # HSV for green: H=60, S=220, V=220 -> RGB: (55, 220, 55)
    frame[170:190, 410:430] = [55, 220, 55]

    res = tracker.process(frame, wrist_img, wrist_3d, t_now=0.0)
    assert res is not None
    assert res.detected is True
    assert res.tip_2d is not None
    assert res.tip_3d is not None
    assert res.quat is not None
    assert len(res.quat) == 4
    # Tip coordinates should have moved forward/right in 3D
    assert res.tip_3d[0] > res.base_3d[0]


def test_prop_tracker_occlusion_fallback():
    tracker = PropTracker(color="green", kind="sword", side="R")
    # Blank black frame (no tape visible)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wrist_img = np.array([0.5, 0.5, 0.0, 1.0])
    wrist_3d = np.array([0.2, 1.0, 0.0])

    # Provide hand quat (pointing straight up)
    from kinetra.core.mathx import IDENTITY_Q
    res = tracker.process(frame, wrist_img, wrist_3d, hand_quat=IDENTITY_Q, t_now=0.0)
    assert res is not None
    assert res.detected is False  # Tape occluded
    assert res.tip_3d is not None  # Fallback orientation resolved!


def test_physics_world_prop_collision_and_sparks():
    physics = PhysicsWorld()
    prop = StuntProp(id=1, kind="box", pos=np.array([0.0, 0.5, 0.0]), size=(1.0, 1.0, 1.0))

    # Fake joint hitting the box at (0.0, 0.5, 0.45)
    joints = np.zeros((33, 4))
    joints[16, :3] = np.array([0.0, 0.5, 0.45])  # Right wrist inside box margin

    physics.test_and_react([prop], joints=joints, prop_res=None)

    assert len(physics.hit_events) > 0
    assert len(physics.particles) > 0  # Sparks spawned!
    # Box received physical velocity impulse
    assert np.linalg.norm(prop.vel) > 0.0

    # Test physics update damping
    v0 = np.linalg.norm(prop.vel)
    physics.update([prop], dt=0.033)
    v1 = np.linalg.norm(prop.vel)
    assert v1 < v0  # Damped by friction!


def test_audio_lipsync_spectral_analysis():
    lipsync = AudioLipSync(enabled=True)

    # Synthetic 400Hz pure tone (simulating vocal vowel 'A')
    sr = 16000
    t = np.linspace(0, 0.032, int(sr * 0.032), endpoint=False)
    vowel_a = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    lipsync._process_audio_chunk(vowel_a)
    bs = lipsync.get_blendshapes()

    assert len(bs) == NBS
    jaw_idx = lipsync._idx["jawOpen"]
    assert bs[jaw_idx] > 0.05  # Jaw opened responding to sound!


def test_multicam_auto_sync():
    auto_sync = MultiCamAutoSync(focal_factor=0.9)
    assert not auto_sync.can_solve()

    # Simulate 20 frames of 2 cameras observing a moving body
    for i in range(25):
        t = i * 0.1
        # 33 landmarks with moving arms
        lm0 = np.full((33, 4), 0.5, dtype=np.float32)
        lm0[15, :2] = [0.3 + 0.1 * np.cos(t), 0.4 + 0.1 * np.sin(t)]
        lm0[16, :2] = [0.7 + 0.1 * np.cos(t), 0.4 - 0.1 * np.sin(t)]

        # Camera 1 from a slight angle
        lm1 = lm0.copy()
        lm1[:, 0] = lm0[:, 0] * 0.9 + 0.05

        auto_sync.add_frame_landmarks(0, lm0)
        auto_sync.add_frame_landmarks(1, lm1)

    assert auto_sync.can_solve()
    rig = auto_sync.solve([(1280, 720), (1280, 720)])
    assert rig is not None
    assert len(rig) == 2
    assert len(rig.cameras) == 2
