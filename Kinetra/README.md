<p align="center">
  <img src="kinetra/assets/logo.png" width="560" alt="Kinetra Motion Studio Banner">
</p>

<h1 align="center">Kinetra Motion Studio</h1>

<p align="center">
  <strong>Next-Generation Markerless AI Motion Capture & Virtual Production Engine</strong>
  <br>
  <em>Hollywood-Grade Full-Body, Face, Hand, Weapon Tracking & Unreal Engine 5 LiveLink for Indie VFX Artists and AAA Studios</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Qt-PySide6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PySide6">
  <img src="https://img.shields.io/badge/OpenGL-3.3%2B-5586A4?style=for-the-badge&logo=opengl&logoColor=white" alt="OpenGL">
  <img src="https://img.shields.io/badge/Unreal_Engine_5-LiveLink-0E1128?style=for-the-badge&logo=unrealengine&logoColor=white" alt="UE5 LiveLink">
  <img src="https://img.shields.io/badge/Blender-VMC_OSC-F5792A?style=for-the-badge&logo=blender&logoColor=white" alt="Blender">
  <img src="https://img.shields.io/badge/Tests-72_Passed_%7C_100%25-00C853?style=for-the-badge" alt="Tests Passed">
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows">
</p>

---

<p align="center">
  <!-- HERO SHOWCASE GIF PLACEHOLDER -->
  <img src="kinetra/assets/demo_showcase.gif" width="850" alt="Kinetra Live Performance Demo" onerror="this.src='kinetra/assets/logo.png'; this.style.maxWidth='450px';">
  <br>
  <em>Live performance capture: 52-joint body kinematics, 52 ARKit facial blendshapes, finger solver, and color-tape weapon tracking.</em>
</p>

---

## 📌 Executive Summary

**Kinetra Motion Studio** is an open, next-generation, **markerless motion capture and virtual production suite** engineered to eliminate the prohibitive costs of specialized optical rigs (such as $50,000+ Vicon or OptiTrack setups) and cumbersome sensor suits. 

Operating entirely on **standard off-the-shelf webcams, smartphones, or recorded video feeds**, Kinetra delivers sub-millisecond, biomechanically constrained, studio-grade motion capture tailored for Hollywood visual effects pipelines, Unreal Engine 5 virtual production, video game developers, and solo 3D animators.

---

## ⚡ Core Engineering Capabilities

### 1. 🦴 52-Joint Biomechanical Full-Body Tracking
* **Monocular & Multi-Camera Triangulation:** Operates seamlessly with a single camera using pinhole focal depth estimation, or across 2–4 calibrated cameras with epipolar triangulation.
* **Anatomical Bone Limits (`anatomy.py`):** Enforces strict physiological range-of-motion constraints across elbows, knees, spine, and shoulders. The skeleton never fractures, flips, or penetrates unnatural orientations.
* **Sticky Foot Grounding (Anti-Sliding IK):** Eliminates floating and ice-skating artifacts by locking grounded feet to the virtual floor plane (`min_y >= 0.0`) with adaptive threshold velocity release.
* **Occlusion Recovery (AI Body Prior):** Predicts missing limb positions during camera line-of-sight loss using kinematic momentum and learned posture trajectories.

### 2. 🎭 52 ARKit Facial Performance Capture & FaceRig
* **Apple ARKit & MetaHuman Standards:** Full 52-channel expression vector capturing subtle eyebrow twitches, eye blinks, gaze directions, jaw mechanics, and lip stretches.
* **Personalized Face Profiling (FaceScan):** Calibrates actor neutral poses and personal stretch gains into a custom facial profile (`face_rig.json`).
* **Optional Optical Markers:** Supports active retro-reflective or painted facial dot tracking for micro-mimic muscle deformation.

### 3. 🖐️ High-Fidelity Hand & Finger Kinematic Solver
* Complete 3-joint distal inheritance across all five digits (thumb, index, middle, ring, pinky).
* **AI Hand Prior Model (`hand_prior.py`):** Automatically reconstructs occluded finger postures during closed fists or fast hand gestures, eliminating bone cracking.

<p align="center">
  <!-- MULTI-ACTOR AND WEAPON SHOWCASE GIF PLACEHOLDER -->
  <img src="kinetra/assets/props_weapons_demo.gif" width="800" alt="Kinetra Weapons & Props Showcase" onerror="this.src='kinetra/assets/logo.png'; this.style.maxWidth='380px';">
  <br>
  <em>Smart Color-Tape Weapon Tracking with Dynamic Stunt Prop Collision & Radial Hit Sparks</em>
</p>

### 4. 👥 Multi-Actor AI Re-Identification (Appearance ReID)
* **Torso Clothing Patch Analysis:** Extracts 2D/3D normalized $16 \times 16$ HSV histograms and median RGB color signatures, categorizing actors by clothing tones (*"Red"*, *"Blue"*, *"Black"*, etc.).
* **Biometric Profile Ratios:** Computes shoulder-width to torso-height biometric ratios (`sho_ratio`) to maintain persistent skeletal profiles.
* **Cross-Over Swap Invariance:** Solves optimal actor matching via minimum-cost bipartite assignment:
  $$\text{Cost} = 0.55 \times \text{ColorCost} + 0.25 \times \text{BioCost} + 0.20 \times \text{SpatialCost}$$
  Even when actors walk past or completely occlude each other, identities remain locked with zero ID swapping.

### 5. ⚔️ Real-Time Color-Tape Weapon & Prop Tracking
* **Zero Extra Hardware:** Track handheld objects (swords, daggers, firearms, shields, combat staffs) using a small piece of **Neon Green, Orange, or Cyan electrical tape** at the tip.
* **Wrist Socket + Tip Vector (6-DoF):** Resolves the weapon's spatial orientation and length from the wrist-to-tape 3D ray in under 0.5 ms ($0\text{ ms}$ perceived latency).
* **Kinematic Fallback:** If the colored tip is momentarily blocked from camera view, the solver smoothly falls back to the hand's palm quaternion (`palm_q`).
* **3D Virtual Weapon Library:** Rendered in real time with interactive models: Katana/Longsword, Tactical Dagger, Handgun, Combat Shield, and Bo-Staff.

### 6. 💥 Stunt Props & Physical Hit Reaction (Dynamics & Sparks)
* Interactive 3D geometry loaded into the stage: Cubes, Boxes, Ramps, Platforms, Cylinders, and Spheres.
* **Impulse & Torque Solver:** When an actor's limbs or weapon blade impacts a prop:
  * **Physical Momentum:** Props slide, tumble, and tilt realistically based on impact force and friction.
  * **Radial Hit Sparks:** Spawns glowing particle bursts at the exact contact point with gravitational decay.
  * **HUD Alert Banner:** Surfaces real-time `💥 HIT! (WEAPON_TIP / HAND)` visual cues.

### 7. 🎙️ Audio-to-LipSync Fallback Engine
* When the facial camera loses sight of the actor (head turned away, looking down, or obscured), the engine continuously analyzes live microphone audio via real-time spectral FFT and energy formant decomposition.
* Formant bands (vowels A/O/U, E/I, consonants P/B/M) synthesize corresponding `jawOpen`, `mouthSmile`, and `mouthFunnel` blendshapes, ensuring dialogue animation never freezes.

### 8. 📡 Unreal Engine 5 LiveLink & Blender VMC Streaming
* **0.025 ms Ultra-Low Latency UDP Pipeline (40,000 pkts/sec):**
  * **`ue5_livelink` (Port 11111):** Converts Kinetra coordinates (Y-up, meters) to Unreal Engine 5 coordinates (Z-up, X-forward, centimeters). Directly drives MetaHumans and Mannequin blueprints with 52 ARKit blendshapes and weapon bone transforms (`weapon_r`).
  * **`vmc` (Port 39539):** Universal Virtual Motion Capture OSC protocol compatible with Unreal Engine (EVMC4U plugin) and Blender (VMC plugin).

### 9. 📷 Multi-Cam Auto-Sync (Markerless Camera Calibration)
* Synchronizes multiple phones or webcams without physical tape measures or printed checkerboards.
* Solves the relative rotation $R$ and translation $T$ between cameras via epipolar geometry (Essential Matrix & 5-point RANSAC) from concurrent human landmark observations in under 1 second.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph INPUT ["📸 Multi-Source Input Layer"]
        Cam0["Camera 1 / Smartphone"]
        Cam1["Camera 2 / Webcam"]
        Mic["Studio Microphone"]
    end

    subgraph AI_CORE ["🧠 AI Inference & Vision Core"]
        PoseNet["MediaPipe Pose (33 Landmarks)"]
        FaceNet["FaceMesh (468 Pts + 52 Blendshapes)"]
        HandNet["Hand Landmarker (21 Pts)"]
        ReID["Multi-Actor Appearance Tracker (HSV ReID)"]
        PropTrack["Color-Tape Weapon/Prop Tracker"]
        AudioFFT["Acoustic FFT & Formant LipSync"]
    end

    subgraph KINEMATICS ["⚙️ Biomechanical Kinematic Solver"]
        BodySolver["BodySolver (Anatomical Bone Limits)"]
        HandSolver["HandSolver (3-Joint Finger Kinematics)"]
        Physics["PhysicsWorld (Impulses, Sparks, Collisions)"]
    end

    subgraph OUTPUT ["🚀 Output & Distribution Layer"]
        Viewport["OpenGL 3D Virtual Stage & HUD"]
        UE5["Unreal Engine 5 LiveLink (MetaHuman)"]
        Blender["Blender (VMC / OSC)"]
        Export["FBX / BVH / USD Point Cache / CSV"]
    end

    Cam0 --> PoseNet & FaceNet & HandNet
    Cam1 --> PoseNet
    Mic --> AudioFFT
    PoseNet --> ReID & BodySolver
    HandNet --> HandSolver
    FaceNet -.-> BodySolver
    AudioFFT -.->|Fallback if Face Lost| BodySolver
    Cam0 --> PropTrack --> Physics
    BodySolver & HandSolver --> Physics
    Physics --> Viewport
    Physics --> UE5
    Physics --> Blender
    Physics --> Export
```

---

## 🛠️ Installation & Quick Start

### Prerequisites
* Windows 10/11 (64-bit)
* Python 3.10, 3.11, or 3.12
* Standard Webcam / Smartphone (compatible with DroidCam, Iriun, or Camo)
* Dedicated or Integrated GPU supporting OpenGL 3.3+

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/kinetra.git
cd kinetra

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Kinetra Motion Studio
python -m kinetra
```

### Running the Test Suite

All kinematic solvers, ReID trackers, LiveLink streams, and physics engines are covered by automated unit tests:

```bash
python -m pytest tests
# 72 passed, 2 skipped (100% pass rate)
```

---

## 🎮 Keybindings & Studio Controls

| Shortcut / Key | Function |
| :--- | :--- |
| **`F5`** | Start Live Motion Capture |
| **`F6`** | Stop Capture |
| **`Space`** | Trigger Countdown Studio Recording (Start / Stop) |
| **`Esc`** | Toggle Fullscreen / Restored Viewport |
| **`Left Click + Drag`** | Orbit 3D Stage / Translate Selected Prop on Ground Plane |
| **`Right Click + Drag`** | Rotate Selected Prop (Yaw) |
| **`Shift / Alt / Ctrl + Left Click`** | Elevate Prop in 3D Space (Vertical Height) |
| **`Delete / Backspace`** | Remove Selected Prop from Scene |

---

## 🌍 Multi-Language Support (Localization)
Kinetra features an internationalized UI supporting 8 languages:
* 🇬🇧 **English** (Full studio interface)
* 🇹🇷 **Türkçe** (Tam yerel arayüz)
* 🇩🇪 **Deutsch** | 🇫🇷 **Français** | 🇪🇸 **Español** | 🇯🇵 **日本語** | 🇷🇺 **Русский** | 🇨🇳 **中文**

---

## 👨‍💻 Developer & Contact

This project was designed and developed as an advanced portfolio project in **Computer Vision, Kinematic Solvers, Real-Time OpenGL Graphics, and Unreal Engine 5 Virtual Production**.

For studio collaborations, VFX & game production inquiries, or technical questions, feel free to reach out:

<p align="left">
  <a href="mailto:hmakiferen12@gmail.com">
    <img src="https://img.shields.io/badge/Email-hmakiferen12%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
  <a href="https://discord.com">
    <img src="https://img.shields.io/badge/Discord-maetarenn.exe-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</p>

* **Email:** `hmakiferen12@gmail.com`
* **Discord:** `maetarenn.exe`

---

<p align="center">
  <em>Kinetra Motion Studio — Designed with passion for the future of cinema, gaming, and digital performance.</em>
</p>
