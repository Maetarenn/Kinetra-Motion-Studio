"""Dynamic Physics, Collision Detection and Hit Reaction for Stage Props and Weapons.

Handles:
1. Physical response (impulses, velocity, friction, bounce) when hit by weapons or limbs.
2. Collision testing between actor joints (wrists, feet, head), weapon tip, and 3D props.
3. Visual hit sparks / impact burst particles.
"""
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HitParticle:
    pos: np.ndarray      # (3,)
    vel: np.ndarray      # (3,)
    color: Tuple[float, float, float, float]
    life: float          # Remaining lifetime [0..1]
    max_life: float


@dataclass
class HitEvent:
    contact_pt: np.ndarray   # (3,)
    normal: np.ndarray       # (3,)
    prop_id: int
    source: str              # "sword", "hand", "foot"
    intensity: float         # 0..1
    timestamp: float


class PhysicsWorld:
    """Simulates physical dynamics and collisions for stage props."""

    GRAVITY = np.array([0.0, -9.81, 0.0])
    FRICTION = 0.88
    RESTITUTION = 0.35  # Bounciness

    def __init__(self):
        self.particles: List[HitParticle] = []
        self.hit_events: List[HitEvent] = []
        self._last_t = time.perf_counter()

    def reset(self):
        self.particles.clear()
        self.hit_events.clear()
        self._last_t = time.perf_counter()

    def update(self, props: list, dt: float = 0.016):
        """Updates physics velocities, damping, and particle lifetimes."""
        # 1. Update Props dynamics
        for p in props:
            if not hasattr(p, "vel"):
                p.vel = np.zeros(3, dtype=np.float64)
            if not hasattr(p, "ang_vel"):
                p.ang_vel = 0.0
            if not hasattr(p, "mass"):
                # Approximate mass from volume (e.g. 15 kg for box)
                p.mass = max(5.0, float(np.prod(p.size)) * 18.0)

            # Velocity integration & damping
            spd = float(np.linalg.norm(p.vel))
            if spd > 1e-4:
                p.pos[0] += float(p.vel[0] * dt)
                p.pos[1] += float(p.vel[1] * dt)
                p.pos[2] += float(p.vel[2] * dt)
                # Damping / friction
                p.vel *= (self.FRICTION ** (dt / 0.016))
                if spd < 0.01:
                    p.vel[:] = 0.0

            if abs(p.ang_vel) > 1e-3:
                p.yaw += float(p.ang_vel * dt)
                p.ang_vel *= (self.FRICTION ** (dt / 0.016))
                if abs(p.ang_vel) < 0.05:
                    p.ang_vel = 0.0

            # Ground clamp
            half_y = p.size[1] / 2.0
            if p.pos[1] < half_y:
                p.pos[1] = half_y
                p.vel[1] = 0.0

        # 2. Update Hit Particles
        active_parts = []
        for pt in self.particles:
            pt.life -= dt / pt.max_life
            if pt.life > 0.0:
                pt.pos += pt.vel * dt
                pt.vel += self.GRAVITY * 0.4 * dt  # Light gravity on sparks
                active_parts.append(pt)
        self.particles = active_parts

        # 3. Clean old hit events (> 0.8s)
        now = time.perf_counter()
        self.hit_events = [e for e in self.hit_events if (now - e.timestamp) < 0.8]

    def spawn_hit_sparks(self, pt: np.ndarray, normal: np.ndarray, count: int = 14,
                         color=(1.0, 0.85, 0.20, 1.0)):
        """Spawns radial spark particles from collision impact."""
        for _ in range(count):
            rand_dir = normal * 0.6 + (np.random.rand(3) - 0.5) * 0.8
            norm = float(np.linalg.norm(rand_dir))
            vel = (rand_dir / max(norm, 1e-4)) * (1.2 + np.random.rand() * 2.2)
            self.particles.append(HitParticle(
                pos=pt.copy(),
                vel=vel,
                color=color,
                life=1.0,
                max_life=0.25 + np.random.rand() * 0.25,
            ))

    def test_and_react(self, props: list, joints: Optional[np.ndarray],
                       prop_res: Optional[object] = None, t_now: float = 0.0):
        """Tests collisions between actor/weapon and props, applying impulses and hit FX."""
        if not props:
            return

        test_points = []

        # 1. Weapon Tip & Blade (highest priority)
        if prop_res is not None and getattr(prop_res, "tip_3d", None) is not None:
            tip = prop_res.tip_3d
            base = prop_res.base_3d
            # Test tip and midpoint of blade
            test_points.append((tip, 0.06, "weapon_tip", 1.8))
            test_points.append(((tip + base) * 0.5, 0.05, "weapon_blade", 1.2))

        # 2. Limbs (Hands, Feet, Head)
        if joints is not None:
            # 15: left wrist, 16: right wrist
            test_points.append((joints[15, :3], 0.09, "hand_L", 0.9))
            test_points.append((joints[16, :3], 0.09, "hand_R", 0.9))
            # 27: left ankle, 28: right ankle
            test_points.append((joints[27, :3], 0.12, "foot_L", 1.4))
            test_points.append((joints[28, :3], 0.12, "foot_R", 1.4))

        # Test against all props
        for prop in props:
            hx, hy, hz = prop.size[0] / 2.0, prop.size[1] / 2.0, prop.size[2] / 2.0
            yaw_rad = math.radians(prop.yaw)
            cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

            for pt, radius, src, force_mult in test_points:
                # Transform pt to prop's local coordinate system
                dx = pt[0] - prop.pos[0]
                dy = pt[1] - prop.pos[1]
                dz = pt[2] - prop.pos[2]

                local_x = dx * cos_y - dz * sin_y
                local_y = dy
                local_z = dx * sin_y + dz * cos_y

                # Clamp to box extents
                cx = np.clip(local_x, -hx, hx)
                cy = np.clip(local_y, -hy, hy)
                cz = np.clip(local_z, -hz, hz)

                dist_sq = (local_x - cx) ** 2 + (local_y - cy) ** 2 + (local_z - cz) ** 2

                # Collision occurred!
                if dist_sq < (radius * radius):
                    dist = math.sqrt(dist_sq)
                    if dist > 1e-4:
                        loc_norm = np.array([(local_x - cx) / dist,
                                             (local_y - cy) / dist,
                                             (local_z - cz) / dist])
                    else:
                        loc_norm = np.array([0.0, 1.0, 0.0])

                    # World space normal
                    world_norm = np.array([
                        loc_norm[0] * cos_y + loc_norm[2] * sin_y,
                        loc_norm[1],
                        -loc_norm[0] * sin_y + loc_norm[2] * cos_y,
                    ])

                    # Physical impulse
                    impulse = -world_norm * (1.6 * force_mult)
                    if not hasattr(prop, "vel"):
                        prop.vel = np.zeros(3, dtype=np.float64)
                    prop.vel += impulse / getattr(prop, "mass", 15.0)

                    # Angular spin reaction if struck off-center
                    if not hasattr(prop, "ang_vel"):
                        prop.ang_vel = 0.0
                    torque = float(local_x * impulse[2] - local_z * impulse[0])
                    prop.ang_vel += torque * 0.4

                    # Spawn Sparks / Hit FX
                    spark_color = (1.0, 0.9, 0.1, 1.0) if "weapon" in src else (0.2, 0.85, 1.0, 1.0)
                    self.spawn_hit_sparks(pt, world_norm, count=16 if "weapon" in src else 8, color=spark_color)

                    self.hit_events.append(HitEvent(
                        contact_pt=pt.copy(),
                        normal=world_norm,
                        prop_id=getattr(prop, "id", 0),
                        source=src,
                        intensity=float(force_mult),
                        timestamp=time.perf_counter(),
                    ))
