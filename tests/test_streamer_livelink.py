"""Unit tests for Unreal Engine 5 LiveLink and VMC real-time streaming."""
import json
import socket
import struct
import time
import numpy as np
import pytest

from kinetra.core.constants import BLENDSHAPE_NAMES
from kinetra.core.exporters import Streamer, KINETRA_TO_UE5_BONES, ARKIT_BS_ORDER
from kinetra.ai.prop_tracker import PropResult


def _parse_arkit_packet(data):
    """Parse a binary ARKit LiveLink v6 packet into a dict."""
    off = 0
    version = data[off]; off += 1
    dev_len = struct.unpack_from('<I', data, off)[0]; off += 4
    device_id = data[off:off+dev_len].decode('utf-8'); off += dev_len
    subj_len = struct.unpack_from('<I', data, off)[0]; off += 4
    subject = data[off:off+subj_len].decode('utf-8'); off += subj_len
    frame, sub, fps, denom = struct.unpack_from('<IIII', data, off); off += 16
    count = data[off]; off += 1
    vals = list(struct.unpack_from(f'<{count}f', data, off))
    # Map values back to named blendshapes
    bs = {}
    for i, name in enumerate(ARKIT_BS_ORDER):
        if i < len(vals):
            bs[name] = vals[i]
    # Extra values beyond 52 (wrinkle slots 52-60)
    extras = vals[len(ARKIT_BS_ORDER):] if len(vals) > len(ARKIT_BS_ORDER) else []
    return {
        'version': version,
        'device_id': device_id,
        'subject': subject,
        'frame': frame,
        'subframe': sub,
        'fps': fps,
        'denominator': denom,
        'count': count,
        'blendshapes': bs,
        'raw_vals': vals,
        'extras': extras,
    }


def test_ue5_livelink_udp_packet():
    # Face socket on main port
    face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    face_sock.bind(("127.0.0.1", 11111))
    face_sock.settimeout(2.0)

    # Body socket on port+1
    body_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    body_sock.bind(("127.0.0.1", 11112))
    body_sock.settimeout(2.0)

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
            length=0.88, confidence=0.9, color_name="Neon Green"
        )

        streamer.send(10, root, quats, faces, prop=prop)

        # ── Verify face packet (binary ARKit v6) ──
        face_data, _ = face_sock.recvfrom(65535)
        pkt = _parse_arkit_packet(face_data)

        assert pkt["version"] == 6
        assert pkt["subject"] == "Kinetra_Face"
        assert pkt["frame"] == 10
        assert pkt["device_id"] == "Kinetra"
        assert pkt["blendshapes"]["eyeBlinkLeft"] == pytest.approx(0.85, abs=0.01)

        # ── Verify body packet (JSON on port+1) ──
        body_data, _ = body_sock.recvfrom(65535)
        body_pkt = json.loads(body_data.decode("utf-8"))

        assert body_pkt["v"] == 1
        assert body_pkt["s"] == "Kinetra_Body"
        assert body_pkt["f"] == 10
        # Check coordinate conversion to UE5 (cm): Z->X, X->Y, Y->Z
        assert body_pkt["r"][0] == pytest.approx(250.0, abs=0.5)   # Z*100
        assert body_pkt["r"][1] == pytest.approx(10.0, abs=0.5)    # X*100
        assert body_pkt["r"][2] == pytest.approx(95.0, abs=0.5)    # Y*100

        # Check bone mappings
        assert "pelvis" in body_pkt["b"]
        assert "spine_01" in body_pkt["b"]
        assert "weapon_r" in body_pkt["b"]  # Live weapon streamed!

    finally:
        streamer.close()
        face_sock.close()
        body_sock.close()


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


def test_face_only_streaming():
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", 11111))
    recv_sock.settimeout(2.0)

    try:
        streamer = Streamer()
        streamer.configure("127.0.0.1", 11111, fmt="ue5_livelink")
        streamer.enabled = True

        faces = [np.zeros(52, dtype=np.float32)]
        faces[0][BLENDSHAPE_NAMES.index("jawOpen")] = 0.75
        wrinkles = {"wrinkle_forehead": 0.65}

        # Face-only: quats is None, root is None
        streamer.send(20, None, None, faces, wrinkles=wrinkles)

        data, _ = recv_sock.recvfrom(65535)
        pkt = _parse_arkit_packet(data)

        assert pkt["version"] == 6
        assert pkt["subject"] == "Kinetra_Face"
        assert pkt["frame"] == 20
        assert pkt["device_id"] == "Kinetra"
        assert pkt["blendshapes"]["jawOpen"] == pytest.approx(0.75, abs=0.01)
        # Wrinkle value should be in the extra slots (indices 52+)
        assert pkt["raw_vals"][52] == pytest.approx(0.65, abs=0.01)

    finally:
        streamer.close()
        recv_sock.close()
