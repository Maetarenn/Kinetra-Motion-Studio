<p align="center">
  <img src="kinetra/assets/logo.png" width="620" alt="Kinetra Motion Studio">
</p>

<p align="center">
  <strong>Next-Generation Markerless AI Motion Capture & Virtual Production Engine</strong>
  <br>
  <em>Hollywood-Grade Full-Body, Face, Hand, Weapon Tracking & Unreal Engine 5 LiveLink for Indie VFX Artists and AAA Studios</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Qt-PySide6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6">
  <img src="https://img.shields.io/badge/OpenGL-3.3%2B-5586A4?style=for-the-badge&logo=opengl&logoColor=white" alt="OpenGL">
  <img src="https://img.shields.io/badge/Unreal_Engine_5-ARKit_LiveLink-0E1128?style=for-the-badge&logo=unrealengine&logoColor=white" alt="UE5 LiveLink">
  <img src="https://img.shields.io/badge/Blender-VMC_OSC-F5792A?style=for-the-badge&logo=blender&logoColor=white" alt="Blender">
  <img src="https://img.shields.io/badge/Tests-80_Passed_%7C_100%25-00C853?style=for-the-badge" alt="Tests Passed">
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows">
</p>

<br>

<p align="center">
  <img src="kinetra/assets/demo_showcase.gif" width="760" alt="Kinetra Live Performance Demo">
  <br>
  <em>Live 60 FPS performance capture: 52-joint body kinematics, 52 ARKit facial blendshapes, finger solver, and real-time streaming.</em>
</p>

<br>

---

## Executive Summary

**Kinetra Motion Studio** is an open, next-generation, **markerless motion capture and virtual production suite** engineered to eliminate the prohibitive costs of specialized optical rigs (such as \$50,000+ Vicon or OptiTrack setups) and cumbersome sensor suits.

Operating entirely on **standard off-the-shelf webcams, smartphones, or recorded video feeds**, Kinetra delivers sub-millisecond, biomechanically constrained, studio-grade motion capture tailored for Hollywood visual effects pipelines, Unreal Engine 5 virtual production, video game developers, and solo 3D animators.

<br>

---

## Core Engineering Capabilities & Features

### 1. 52-Joint Biomechanical Full-Body Kinematics (`body_solver.py`, `anatomy.py`)
* **Monocular & Multi-Camera Triangulation:** Operates seamlessly with a single camera using pinhole focal depth estimation, or across 2-20+ calibrated cameras with epipolar triangulation.
* **Anatomical Bone Limits:** Enforces strict physiological range-of-motion constraints across elbows, knees, spine, and shoulders. The skeleton never fractures, flips, or penetrates unnatural orientations.
* **Sticky Foot Grounding (Anti-Foot Sliding IK):** Eliminates floating and ice-skating artifacts by locking grounded feet to the virtual floor plane ($Y \ge 0.0$) with adaptive threshold velocity release.
* **Occlusion Recovery (AI Body Prior):** Predicts missing limb positions during camera line-of-sight loss using kinematic momentum and learned posture trajectories.

### 2. Native C++17 Dynamic Physics Engine & LBS (`kinetra_physics.dll`)
* **Dynamic Physics Constraint Solver:** Written in native C++17 with MSVC x64 SIMD optimizations. Employs a mass-spring-damper (PD) network to eliminate high-frequency camera noise and joint jitter.
* **Fast Linear Blend Skinning (`kinetra_fast_lbs`):** Deforms high-poly 3D meshes in real time at **200+ FPS** without CPU bottlenecks.
* **Zero-Downtime Fallback:** Connected via `ctypes` bridge with an automatic pure NumPy fallback if the compiled DLL is absent.

### 3. Unskeletized 3D Mesh Auto-Rigging Engine (`auto_rigger.py`)
* **Instant Character Import:** Drag & drop raw, bone-free `.obj` and `.glb`/`.gltf` 3D character meshes (humanoids, robots, creatures) directly into the viewport.
* **Automatic Spatial Alignment:** Normalizes mesh scale to human height, centers root origin, and snaps foot soles to the stage ground plane ($Y = 0$).
* **Volumetric Distance-Field Skinning:** Computes smooth bone skinning weights (top 4 influences per vertex) with strict anatomical left/right arm isolation to prevent cross-limb skin tearing.

### 4. 52 ARKit Facial Performance Capture & FaceScan (`facerig.py`)
* **Apple ARKit & MetaHuman Standards:** Full 52-channel expression vector capturing subtle eyebrow twitches, eye blinks, gaze directions, jaw mechanics, and lip stretches.
* **Guided Personal Profiling (FaceScan):** Calibrates actor neutral poses, pupil offsets, and muscle stretch gains into a custom facial profile (`face_rig.json`).
* **Optical Marker Support:** Supports active retro-reflective or painted facial dot tracking for micro-mimic muscle deformation.

### 5. High-Fidelity Hand & 3-Joint Finger Solver (`hand.py`, `hand_prior.py`)
* Complete 3-joint distal kinematic chain across all five digits (thumb, index, middle, ring, pinky).
* **AI Hand Prior Model:** Automatically reconstructs occluded finger postures during closed fists or fast hand gestures, eliminating bone cracking.

<br>

<p align="center">
  <img src="kinetra/assets/props_weapons_demo.gif" width="760" alt="Kinetra Weapons & Props Showcase">
  <br>
  <em>Smart Color-Tape Weapon Tracking with Dynamic Stunt Prop Collision & Radial Hit Sparks</em>
</p>

<br>

### 6. Real-Time Color-Tape Weapon & Prop Tracking (`prop_tracker.py`)
* **Zero Extra Hardware:** Track handheld combat props (swords, katanas, daggers, handguns, shields, combat staffs) using a small piece of **Neon Green, Orange, or Cyan electrical tape** at the tip.
* **Wrist Socket + Tip Vector (6-DoF):** Resolves the weapon's spatial orientation and length from the wrist-to-tape 3D ray in under 0.5 ms.
* **Kinematic Fallback:** If the colored tip is momentarily blocked from camera view, the solver smoothly falls back to the hand's palm quaternion (`palm_q`).
* **3D Virtual Weapon Library:** Rendered in real time with interactive models: Katana/Longsword, Tactical Dagger, Handgun, Combat Shield, and Bo-Staff.

### 7. Stunt Props & Physical Hit Reaction (`physics_world.py`)
* Interactive 3D geometry loaded into the stage: Cubes, Boxes, Ramps, Platforms, Cylinders, and Spheres.
* **Impulse & Torque Solver:** When an actor's limbs or weapon blade impacts a prop:
  * **Physical Momentum:** Props slide, tumble, and tilt realistically based on impact force and friction.
  * **Radial Hit Sparks:** Spawns glowing particle bursts at the exact contact point with gravitational decay.
  * **HUD Alert Banner:** Surfaces real-time visual hit indicators.

### 8. Multi-Actor AI Re-Identification (`actor_tracker.py`)
* **Torso Clothing Patch Analysis:** Extracts 2D/3D normalized $16 \times 16$ HSV histograms and median RGB color signatures, categorizing actors by clothing tones (*"Red"*, *"Blue"*, *"Black"*).
* **Biometric Profile Ratios:** Computes shoulder-width to torso-height biometric ratios to maintain persistent skeletal profiles.
* **Cross-Over Swap Invariance:** Solves optimal actor matching via minimum-cost bipartite assignment:
  $$\text{Cost} = 0.55 \times \text{ColorCost} + 0.25 \times \text{BioCost} + 0.20 \times \text{SpatialCost}$$
  Even when actors walk past or completely occlude each other, identities remain locked with zero ID swapping.

### 9. Audio-to-LipSync Fallback Engine (`audio_lipsync.py`)
* When the facial camera loses sight of the actor (head turned away, looking down, or obscured), the engine continuously analyzes live microphone audio via real-time spectral FFT and energy formant decomposition.
* Formant bands (vowels A/O/U, E/I, consonants P/B/M) synthesize corresponding `jawOpen`, `mouthSmile`, and `mouthFunnel` blendshapes, ensuring dialogue animation never freezes.

### 10. Multi-Cam Auto-Sync Calibration (`multicam_sync.py`)
* Synchronizes multiple phones or webcams without physical tape measures or printed checkerboards.
* Solves the relative rotation $R$ and translation $T$ between cameras via epipolar geometry (Essential Matrix & 5-point RANSAC) from concurrent human landmark observations in under 1 second.

### 11. iPhone TrueDepth / LiDAR Precision Face Capture (`iphone_receiver.py`)
* **Micron-Precision Face & Wrinkle Capture:** Leverages iPhone's TrueDepth and LiDAR sensors via **Live Link Face** or **iFacialMocap** apps for industry-leading facial capture accuracy.
* **Dual Protocol Support:** Accepts both Apple Live Link Face JSON packets and iFacialMocap binary format on a configurable UDP port (default: 49983).
* **Real-Time Wrinkle Injection:** Automatically routes iPhone-captured micro-wrinkle data (forehead creases, nasolabial folds, crow's feet) into the wrinkle solver pipeline for ultra-realistic facial performance.
* **Seamless Multi-Source Fusion:** iPhone face data can run alongside webcam body capture for simultaneous full-body + precision face tracking.

### 12. Micro-Wrinkle Simulation Engine (`wrinkle_solver.py`)
* **9-Channel Wrinkle System:** Computes real-time micro-wrinkle deformations from blendshape combinations: forehead creases, frown furrows, crow's feet, nasolabial folds, lip compression, chin dimple, nose bridge, and marionette lines.
* **MetaHuman-Compatible Wrinkle Curves:** Outputs `get_metahuman_curves()` dictionary directly compatible with Unreal Engine 5 MetaHuman wrinkle map system.
* **Blendshape-Driven:** All wrinkles are derived algorithmically from the standard 52 ARKit blendshapes -- no additional hardware or sensors required (enhanced further when paired with iPhone LiDAR).

### 13. N-Camera Studio Support (Up to 32 Devices) (`cameras.py`, `camera_dialog.py`)
* **Unlimited Camera Scaling:** Connect up to 32 cameras simultaneously (webcams, USB capture cards, IP cameras, smartphones via DroidCam/Iriun/Camo) for maximum triangulation accuracy and coverage.
* **Camera Setup Manager Dialog:** Professional multi-camera management UI with per-camera role assignment (Main, Side, Top, Back, Detail, Extra, Custom), IP/RTSP stream support, live preview, and one-click device rescanning.
* **Auto Epipolar Sync:** Calibrate all cameras without printed chessboards using human landmark-based epipolar geometry.

### 14. Unreal Engine 5 Native ARKit LiveLink & Blender VMC Streaming (`exporters.py`)
* **Native ARKit Binary Protocol (v6):** Face blendshapes are streamed as native Apple ARKit LiveLink binary packets, directly compatible with UE5's built-in **Apple ARKit** LiveLink source -- zero custom plugins needed.
  * **Port 11111:** 61-channel ARKit face data (52 standard + 9 wrinkle channels) at native binary speed.
  * **Port 11112:** Body skeleton data as JSON with full UE5 MetaHuman/Mannequin bone mapping and coordinate system conversion (Y-up meters to Z-up centimeters).
* **`vmc` (Port 39539):** Universal Virtual Motion Capture OSC protocol compatible with Unreal Engine (EVMC4U plugin) and Blender (VMC addon).
* **Client Auto-Detection (`client_detector.py`):** Automatically detects whether Unreal Engine 5 or Blender is running on the target machine and displays real-time connection status in the UI.
* **Blender Receiver Addon (`kinetra_blender_receiver.py`):** Drop-in Blender Python addon that receives Kinetra's live data and drives armature bones in real time.

### 15. Studio Recording, Takes Manager & Multi-Format Export
* Multi-take recording system with countdown timer and Take Manager.
* Non-destructive take review, playback scrubbing, and clipping.
* **Industry-Standard Exports:** BVH, FBX, USD Point Cache, CSV, and raw NumPy data.

### 16. Professional UI & Icon System (`icons.py`, `theme.py`)
* **QPainter Vector Icon Factory:** 16+ hand-crafted vector icons (camera, scan, play, stop, record, calibrate, face, wave, dots, folder, arrows, crosshair, add, remove, refresh, preview, setup) rendered at any DPI without pixelation.
* **Resizable Split Panel:** Draggable splitter handle with hover feedback for adjustable parameter/viewport layout.
* **Full i18n Camera Dialog:** All camera management strings are localized across all 8 supported languages.

<br>

---

## System Architecture

```mermaid
flowchart TD
    subgraph INPUT ["Multi-Source Input Layer"]
        Cam0["Camera 1-32 / Smartphones"]
        iPhone["iPhone TrueDepth / LiDAR"]
        Mic["Studio Microphone"]
    end

    subgraph AI_CORE ["AI Inference & Vision Core"]
        PoseNet["MediaPipe Pose (33 Landmarks)"]
        FaceNet["FaceMesh (468 Pts + 52 Blendshapes)"]
        HandNet["Hand Landmarker (21 Pts)"]
        ReID["Multi-Actor Appearance Tracker (HSV ReID)"]
        PropTrack["Color-Tape Weapon/Prop Tracker"]
        AudioFFT["Acoustic FFT & Formant LipSync"]
        iPhoneRcv["iPhone LiDAR Receiver (UDP)"]
    end

    subgraph KINEMATICS ["Biomechanical Kinematic Solver"]
        BodySolver["BodySolver (Anatomical Bone Limits)"]
        HandSolver["HandSolver (3-Joint Finger Kinematics)"]
        WrinkleSolver["WrinkleSolver (9-Channel Micro-Wrinkles)"]
        Physics["PhysicsWorld (Impulses, Sparks, Collisions)"]
        CPP["Native C++ Dynamic Physics Engine"]
    end

    subgraph OUTPUT ["Output & Distribution Layer"]
        Viewport["OpenGL 3D Virtual Stage & HUD"]
        UE5["UE5 ARKit LiveLink (Binary v6)"]
        Blender["Blender (VMC / OSC)"]
        Export["FBX / BVH / USD Point Cache / CSV"]
    end

    Cam0 --> PoseNet & FaceNet & HandNet
    iPhone --> iPhoneRcv --> WrinkleSolver
    Mic --> AudioFFT
    PoseNet --> ReID & BodySolver
    HandNet --> HandSolver
    FaceNet -.-> BodySolver
    FaceNet --> WrinkleSolver
    AudioFFT -.->|Fallback if Face Lost| BodySolver
    Cam0 --> PropTrack --> Physics
    BodySolver & HandSolver --> CPP --> Physics
    WrinkleSolver --> Physics
    Physics --> Viewport
    Physics --> UE5
    Physics --> Blender
    Physics --> Export
```

<br>

---

## Installation & Quick Start

### Option A: Standalone Windows Installer (Recommended)
Download **`Kinetra_Setup_v0.4.0.exe`** from the **[Releases](../../releases)** tab. Double-click to launch the wizard, select your Mocap Takes directory, and start capturing immediately -- no Python or dependencies required!

### Option B: From Source Code

```bash
# 1. Clone the repository
git clone https://github.com/maetarenn/kinetra.git
cd kinetra

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Kinetra Motion Studio
python run_kinetra.py
```

### Running Automated Test Suite

```bash
python -m pytest tests
# 80 passed, 2 skipped (100% pass rate in ~67s)
```

<br>

---

## Keybindings & Studio Controls

| Shortcut / Key | Function |
| :--- | :--- |
| **`F5`** | Start Live Motion Capture |
| **`F6`** | Stop Capture |
| **`Space`** | Trigger Countdown Studio Recording (Start / Stop) |
| **`Esc`** | Toggle Fullscreen / Restored Viewport |
| **`Left Click + Drag`** | Orbit 3D Stage / Translate Selected Prop on Ground Plane |
| **`Right Click + Drag`** | Rotate Selected Prop (Yaw) |
| **`Shift / Alt / Ctrl + Drag`** | Elevate Prop in 3D Space (Vertical Height) |
| **`Delete / Backspace`** | Remove Selected Prop from Scene |

<br>

---

## Multi-Language Support (Localization)
Kinetra features a fully internationalized UI supporting **8 languages** with complete key parity across all locale files:

| Language | Status |
| :--- | :--- |
| English | Full studio interface (262 keys) |
| Turkce | Tam yerel arayuz (262 keys) |
| Deutsch | Vollstandige Oberflache |
| Francais | Interface complete |
| Espanol | Interfaz completa |
| Japanese | Full interface |
| Russian | Full interface |
| Chinese (Simplified) | Full interface |

<br>

---

## Developer & Contact

This project was designed and engineered as a comprehensive showcase in **Computer Vision, Kinematic Solvers, Real-Time OpenGL Graphics, Physics Simulations, and Unreal Engine 5 Virtual Production**.

For studio collaborations, VFX & game production inquiries, or technical questions, feel free to reach out:

<p align="left">
  <a href="mailto:hmakiferen12@gmail.com">
    <img src="https://img.shields.io/badge/Email-hmakiferen12%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
  <a href="https://discord.com">
    <img src="https://img.shields.io/badge/Discord-maetarenn.exe-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</p>

* **Email:** [hmakiferen12@gmail.com](mailto:hmakiferen12@gmail.com)
* **Discord:** `maetarenn.exe`

<br>

---

<p align="center">
  <em>Kinetra Motion Studio -- Designed with passion for the future of cinema, gaming, and digital performance.</em>
</p>
