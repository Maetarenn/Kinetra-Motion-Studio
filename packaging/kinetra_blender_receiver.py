"""Kinetra Motion Studio -> Blender Real-Time UDP Live Receiver Script.

HOW TO USE IN BLENDER (3.0 - 4.x):
1. In Blender, switch to the 'Scripting' workspace.
2. Click 'New', paste this script, and click the '▶ Run Script' button (or Alt+P).
3. Select your character's Armature in the 3D Viewport.
4. In Kinetra, select 'json' format (Port 9000) and check 'Stream' / 'Canli Yayin'.
Your character in Blender will now move and animate in real-time at 60 FPS!
Press ESC in Blender 3D Viewport to stop live streaming.
"""
import bpy
import json
import socket
import mathutils

PORT = 9000  # Kinetra JSON UDP Port (use 'json' format in Kinetra)

# Mapping from Kinetra/UE5 bones to Blender standard humanoid bone names
BONE_MAPPING = {
    "pelvis": "Hips",
    "spine_01": "Spine",
    "spine_02": "Spine1",
    "spine_03": "Spine2",
    "neck_01": "Neck",
    "head": "Head",
    "upperarm_l": "LeftUpperArm",
    "lowerarm_l": "LeftLowerArm",
    "hand_l": "LeftHand",
    "upperarm_r": "RightUpperArm",
    "lowerarm_r": "RightLowerArm",
    "hand_r": "RightHand",
    "thigh_l": "LeftUpperLeg",
    "calf_l": "LeftLowerLeg",
    "foot_l": "LeftFoot",
    "thigh_r": "RightUpperLeg",
    "calf_r": "RightLowerLeg",
    "foot_r": "RightFoot",
}


class KinetraLiveReceiverOperator(bpy.types.Operator):
    """Real-Time Kinetra Motion Capture Streamer for Blender."""
    bl_idname = "wm.kinetra_live_receiver"
    bl_label = "Kinetra Live MoCap Receiver"

    _timer = None
    _sock = None

    def modal(self, context, event):
        if event.type == 'ESC':
            self.cancel(context)
            self.report({'INFO'}, "Kinetra Live Stream Durduruldu.")
            return {'FINISHED'}

        if event.type == 'TIMER':
            self.process_packets(context)

        return {'PASS_THROUGH'}

    def process_packets(self, context):
        if not self._sock:
            return

        latest_data = None
        while True:
            try:
                data, _ = self._sock.recvfrom(65535)
                latest_data = data
            except (socket.error, BlockingIOError):
                break

        if not latest_data:
            return

        try:
            pkt = json.loads(latest_data.decode('utf-8'))
        except Exception:
            return

        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            return

        # 1. Update Root Translation
        root = pkt.get("root")
        if root and len(root) == 3:
            # Scale from UE5 cm or Kinetra m to Blender units
            scale = 0.01 if max(abs(root[0]), abs(root[1]), abs(root[2])) > 20.0 else 1.0
            obj.location = (root[0] * scale, root[1] * scale, root[2] * scale)

        # 2. Update Bone Rotations
        bones_data = pkt.get("bones", {})
        armature = obj.pose
        if not armature:
            return

        for ue_bone, quat_vals in bones_data.items():
            if len(quat_vals) != 4:
                continue
            # Look up bone name
            target_bone_name = BONE_MAPPING.get(ue_bone.lower(), ue_bone)
            pose_bone = armature.bones.get(target_bone_name) or armature.bones.get(ue_bone)
            if pose_bone:
                pose_bone.rotation_mode = 'QUATERNION'
                # quat_vals: [w, x, y, z]
                pose_bone.rotation_quaternion = mathutils.Quaternion((quat_vals[0], quat_vals[1], quat_vals[2], quat_vals[3]))

        # 3. Update Shape Keys (ARKit Blendshapes for Facial Animation)
        blendshapes = pkt.get("blendshapes", {})
        if blendshapes:
            for child in obj.children:
                if child.type == 'MESH' and child.data.shape_keys:
                    kb = child.data.shape_keys.key_blocks
                    for bs_name, bs_val in blendshapes.items():
                        if bs_name in kb:
                            kb[bs_name].value = float(bs_val)

    def execute(self, context):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("0.0.0.0", PORT))
            self._sock.setblocking(False)
        except Exception as e:
            self.report({'ERROR'}, f"Port {PORT} açılamadı: {e}")
            return {'CANCELLED'}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)
        self.report({'INFO'}, f"Kinetra Live Stream Başlatıldı! (Port {PORT} dinleniyor, durdurmak için ESC)")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self._sock:
            self._sock.close()
            self._sock = None


def register():
    bpy.utils.register_class(KinetraLiveReceiverOperator)
    # Auto-start only if a window context is available
    try:
        bpy.ops.wm.kinetra_live_receiver('INVOKE_DEFAULT')
    except RuntimeError:
        pass  # No active window context (headless mode)


def unregister():
    bpy.utils.unregister_class(KinetraLiveReceiverOperator)


if __name__ == "__main__":
    register()
