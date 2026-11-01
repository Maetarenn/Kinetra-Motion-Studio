# Kinetra Motion Studio - Release Notes

---

## v0.4.0 (2026-11-01)

### Highlights
Kinetra v0.4.0 is a major feature release that transforms the studio into a **production-grade multi-camera capture pipeline** with native Unreal Engine 5 integration, iPhone LiDAR precision face capture, and a micro-wrinkle simulation engine. This release also delivers a complete UI overhaul with professional vector icons and full internationalization.

---

### New Features

#### iPhone TrueDepth / LiDAR Precision Face Capture
- **`iphone_receiver.py`** -- Dedicated UDP receiver for iPhone face data via Live Link Face and iFacialMocap apps.
- Dual protocol support: Apple Live Link Face JSON and iFacialMocap binary format.
- Configurable UDP port (default: 49983) with automatic packet format detection.
- Seamless fusion with webcam body capture for simultaneous full-body + precision face.
- UI checkbox with tooltip in the Live Output panel for one-click iPhone integration.

#### Micro-Wrinkle Simulation Engine
- **`wrinkle_solver.py`** -- Real-time 9-channel wrinkle deformation system.
- Channels: forehead creases, frown furrows, crow's feet, nasolabial folds, lip compression, chin dimple, nose bridge, and marionette lines.
- All wrinkles derived algorithmically from 52 ARKit blendshapes.
- `get_metahuman_curves()` API for direct UE5 MetaHuman wrinkle map compatibility.
- Enhanced precision when paired with iPhone LiDAR input.

#### N-Camera Studio Support (Up to 32 Devices)
- Camera scan range expanded from 6 to **32 devices** (`cameras.py`).
- Support for USB webcams, capture cards, IP cameras (RTSP/HTTP), and smartphone bridges (DroidCam, Iriun, Camo).
- Studio-grade multi-angle triangulation for maximum accuracy and occlusion recovery.

#### Camera Setup Manager Dialog
- **`camera_dialog.py`** -- Complete rewrite with professional styling.
- Per-camera role assignment: Main, Side, Top, Back, Detail, Extra, Custom.
- IP/RTSP stream toggle per camera with URL input and tooltip.
- Live camera preview with start/stop capability.
- One-click device rescanning across all 32 indices.
- Auto Epipolar Sync checkbox for chessboard-free calibration.
- Gradient row styling, hover effects, styled scrollbars, and dark theme integration.
- **Full i18n:** All dialog strings use the `t()` localization function (27 new locale keys).

#### Native ARKit Binary Protocol for UE5 LiveLink
- **`exporters.py`** -- UE5 LiveLink now sends native Apple ARKit binary packets (protocol v6).
- Face data on port 11111: 61 channels (52 ARKit + 9 wrinkle), directly compatible with UE5's built-in Apple ARKit LiveLink source.
- Body skeleton data on port 11112 (port+1) as JSON with full MetaHuman/Mannequin bone mapping.
- Coordinate system conversion: Kinetra Y-up meters to UE5 Z-up X-forward centimeters.
- `ARKIT_BS_ORDER` (51 standard Apple blendshapes) and `_arkit_mapping()` for runtime blendshape name resolution.
- Zero UE5 plugin required -- works out of the box with the built-in Apple ARKit source.

#### Client Auto-Detection
- **`client_detector.py`** -- Detects running Unreal Engine 5 and Blender instances.
- Real-time connection status displayed in the UI with localized strings.
- Validates that UE5 is actually installed and running before reporting "connected."

#### Professional Vector Icon System
- **`icons.py`** -- QPainter-based icon factory with 16+ vector icons.
- Icons: camera, scan, play, stop, record, calibrate, face, wave, dots, folder, arrows, crosshair, add, remove, refresh, preview, setup.
- Resolution-independent rendering -- crisp at any DPI without rasterization artifacts.
- All emojis removed from the entire codebase and replaced with professional icons.

#### Blender Receiver Addon
- **`kinetra_blender_receiver.py`** -- Drop-in Blender addon for real-time animation.
- Receives Kinetra's live JSON data on UDP port 9000.
- Drives armature bones in real time within the Blender viewport.
- Safe headless-mode invocation with error handling (no crash in background/batch scripts).

---

### Improvements

#### UI / UX
- Resizable left panel: QSplitter handle made visible (5px) with cyan hover effect.
- Left panel minimum width reduced from 310px to 240px for more viewport space.
- Camera Setup button repositioned to its own row (no longer overlaps the Scan button).
- All deck control buttons now display professional vector icons.
- Stream guide button bound to i18n key for proper localization.

#### Internationalization (i18n)
- 27 new locale keys added across all 8 language files (en, tr, de, es, fr, ja, ru, zh).
- Camera dialog, iPhone checkbox, stream status, and button labels fully localized.
- All locale files maintain complete key parity (verified by automated test).
- Emoji characters removed from all locale `export.btn_open_folder` values.

#### Streaming & Export
- `KINETRA_TO_UE5_BONES` mapping table: 52 Kinetra bone names to UE5 MetaHuman/Mannequin standard names.
- Dual-socket streaming architecture: separate sockets for face (binary) and body (JSON) data.
- Weapon bone (`weapon_r` / `weapon_l`) included in body stream packets.

---

### Bug Fixes
- **UE5 LiveLink was sending JSON instead of native binary** -- UE5's Apple ARKit source expects binary protocol v6, not JSON. Fixed by implementing the complete ARKit binary packet builder.
- **Blender receiver addon crashed in headless mode** -- Auto-invoke wrapped in `try/except RuntimeError` to prevent crash when Blender runs without a UI context.
- **Camera scan limited to 6 devices** -- Expanded `max_index` from 6 to 32 for studio-scale camera arrays.
- **Camera dialog stayed in Turkish regardless of language setting** -- All hardcoded Turkish strings replaced with `t()` locale lookups.
- **Camera Setup button overlapped the Scan button** -- Moved to a dedicated row in the form layout.
- **Emojis used in UI instead of professional icons** -- All emoji characters removed and replaced with QPainter vector icons.
- **Client status reported UE5 as connected without checking if it was installed** -- `client_detector.py` now validates running processes.
- **`len(sources)` bug in main_window.py** -- Fixed to `len(dlg.selected_sources)` for correct camera count after dialog selection.

---

### Test Suite
- **80 passed, 2 skipped** (up from 72 passed in v0.3.x).
- 8 new test cases covering wrinkle solver, iPhone receiver, client detector, N-camera rig, and updated LiveLink binary protocol tests.
- `test_ue5_livelink_udp_packet` -- Now validates binary ARKit v6 face packets + JSON body packets on dual ports.
- `test_face_only_streaming` -- Validates face-only binary streaming with wrinkle data injection.
- `test_all_locales_complete_and_placeholders_match` -- Enforces key parity across all 8 locale files.
- Full test suite completes in ~67 seconds.

---

### New Files
| File | Description |
| :--- | :--- |
| `kinetra/core/wrinkle_solver.py` | 9-channel micro-wrinkle simulation engine |
| `kinetra/core/client_detector.py` | UE5 / Blender process auto-detection |
| `kinetra/capture/iphone_receiver.py` | iPhone TrueDepth/LiDAR UDP face receiver |
| `kinetra/ui/icons.py` | Professional QPainter vector icon factory |
| `kinetra/ui/camera_dialog.py` | Multi-camera setup manager dialog |
| `packaging/kinetra_blender_receiver.py` | Blender real-time receiver addon |
| `tests/test_wrinkles_and_iphone.py` | Tests for wrinkle solver, iPhone receiver, client detector |

---

### Dependencies
- Added `psutil>=5.9` for process detection (client detector).
- No other new dependencies.

---

### Migration Notes
- **UE5 Users:** LiveLink source type must be set to **Apple ARKit** (not generic JSON). Face data arrives on the configured port (default 11111), body data on port+1 (default 11112).
- **Blender Users:** Install `kinetra_blender_receiver.py` as a Blender addon. The receiver now listens on port 9000 (changed from 11111 to avoid conflict with UE5 face data).
- **Multi-Camera Users:** The Camera Setup Manager dialog is now the primary way to configure cameras. Access it via the "Camera Setup" button in the left panel.
