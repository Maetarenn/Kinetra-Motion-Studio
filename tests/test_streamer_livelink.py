"""Unit tests for Unreal Engine 5 LiveLink and VMC real-time streaming."""
import json
import socket
import time
import numpy as np
import pytest

from kinetra.core.constants import BLENDSHAPE_NAMES
from kinetra.core.exporters import Streamer, KINETRA_TO_UE5_BONES
from kinetra.ai.prop_tracker import PropResult


def test_ue5_livelink_udp_packet():
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 11111))
    recv_sock.settimeout(2.0)

    try:
        streamer = Streamer()
        streamer.configure("127.0.0.1", 11111, fmt="ue5_livelink")
        streamer.enabled = True

        root = np.array([0.1, 0.95, 2.5])  # meters
        quats = {
            "Hips": np.array([1.0, 0.0, 0.0, 0.0]),
            "spine": np.array([0.98, 0.1, 0.0, 0.0]),
            "head": np.array([0.95, 0.0, 0.2, 0.0]),
        }
        faces = [np.zeros(52, dtype=np.float32)]
        faces[0][BLENDSHAPE_NAMES.index("eyeBlinkLeft")] = 0.85

        prop = PropResult(
            detected=True, kind="sword", side="R",
            tip_2d=(0.5, 0.3), tip_3d=np.array([0.5, 1.2, 2.8]),
            base_3d=np.array([0.3, 1.0, 2.5]), quat=np.array([0.707, 0.0, 0.707, 0.0]),
            length=0.88, confidence=0.9, color_name="Neon Yeşil"
        )

        streamer.send(10, root, quats, faces, prop=prop)

        data, _ = recv_sock.recvfrom(65535)
        pkt = json.loads(data.decode("utf-8"))

        assert pkt["version"] == 1
        assert pkt["subject"] == "Kinetra_Actor"
        assert pkt["frame"] == 10
        # Check coordinate conversion to UE5 (cm)
        assert pkt["root"][0] == pytest.approx(250.0, abs=0.5)  # Z -> X (Forward)
        assert pkt["root"][1] == pytest.approx(10.0, abs=0.5)   # X -> Y (Right)
        assert pkt["root"][2] == pytest.approx(95.0, abs=0.5)   # Y -> Z (Up)

        # Check bone mappings
        assert "pelvis" in pkt["bones"]
        assert "spine_01" in pkt["bones"]
        assert "weapon_r" in pkt["bones"]  # Live weapon streamed!

        # Check ARKit blendshapes
        assert pkt["blendshapes"]["eyeBlinkLeft"] == pytest.approx(0.85, abs=0.01)
        assert pkt["blendshapes"]["EyeBlinkLeft"] == pytest.approx(0.85, abs=0.01)

    finally:
        streamer.close()
        recv_sock.close()


def test_vmc_protocol_bundle():
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 39539))
    recv_sock.settimeout(2.0)

    try:
        streamer = Streamer()
        streamer.configure("127.0.0.1", 39539, fmt="vmc")
        streamer.enabled = True

        root = np.array([0.0, 1.0, 2.0])
        quats = {"Hips": [1.0, 0.0, 0.0, 0.0], "head": [0.9, 0.1, 0.0, 0.0]}
        faces = [np.zeros(52, dtype=np.float32)]

        streamer.send(1, root, quats, faces)

        data, _ = recv_sock.recvfrom(65535)
        assert data.startswith(b"#bundle")
        assert b"/VMC/Ext/Root/Pos" in data
        assert b"/VMC/Ext/Bone/Hips" in data
        assert b"/VMC/Ext/OK" in data
    finally:
        streamer.close()
        recv_sock.close()
