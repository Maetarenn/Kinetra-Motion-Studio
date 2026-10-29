"""Shared constants: joint ids, ARKit blendshape names, export skeleton layout."""
HIP_MID, SHO_MID, EAR_MID = 33, 34, 35          # synthetic joints appended after the 33 MediaPipe ones
NOSE = 0
EAR_L, EAR_R = 7, 8
SHO_L, SHO_R = 11, 12
HIP_L, HIP_R = 23, 24
FOOT_IDS = (27, 28, 29, 30, 31, 32)

# side -> (proximal, hinge, distal, dependents that travel with the distal joint)
ARMS = {"L": (11, 13, 15, (17, 19, 21)), "R": (12, 14, 16, (18, 20, 22))}
LEGS = {"L": (23, 25, 27, (29, 31)), "R": (24, 26, 28, (30, 32))}
HEEL = {"L": 29, "R": 30}
TOE = {"L": 31, "R": 32}

BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]
BS_INDEX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}
NBS = len(BLENDSHAPE_NAMES)
assert NBS == 52

POSE_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
                    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
                    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32))

# Export skeleton (BVH / USD). Parents precede children.
JOINT_ORDER = ("Hips", "spine", "head", "upper_arm_L", "forearm_L", "upper_arm_R", "forearm_R",
               "thigh_L", "shin_L", "foot_L", "thigh_R", "shin_R", "foot_R")
JOINT_PARENT = {"spine": "Hips", "head": "spine", "upper_arm_L": "spine", "forearm_L": "upper_arm_L",
                "upper_arm_R": "spine", "forearm_R": "upper_arm_R", "thigh_L": "Hips", "shin_L": "thigh_L",
                "foot_L": "shin_L", "thigh_R": "Hips", "shin_R": "thigh_R", "foot_R": "shin_R"}
DEFAULT_LENGTHS = {"spine": .50, "head": .22, "sw": .19, "hw": .10, "upper_arm": .30, "forearm": .26,
                   "thigh": .43, "shin": .42, "foot": .15}


def skeleton_offsets(L):
    """Rest-pose joint offsets in metres relative to the parent joint (y up, +x = character's left)."""
    off = {"Hips": (0, 0, 0), "spine": (0, 0, 0), "head": (0, L["spine"], 0),
           "upper_arm_L": (L["sw"], L["spine"], 0), "forearm_L": (0, -L["upper_arm"], 0),
           "upper_arm_R": (-L["sw"], L["spine"], 0), "forearm_R": (0, -L["upper_arm"], 0),
           "thigh_L": (L["hw"], 0, 0), "shin_L": (0, -L["thigh"], 0), "foot_L": (0, -L["shin"], 0),
           "thigh_R": (-L["hw"], 0, 0), "shin_R": (0, -L["thigh"], 0), "foot_R": (0, -L["shin"], 0)}
    end = {"head": (0, L["head"], 0), "forearm_L": (0, -L["forearm"], 0), "forearm_R": (0, -L["forearm"], 0),
           "foot_L": (0, 0, L["foot"]), "foot_R": (0, 0, L["foot"])}
    return off, end
