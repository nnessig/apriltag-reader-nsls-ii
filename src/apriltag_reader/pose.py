"""Pose math and temporal pose filtering."""

import math

import numpy as np


def rotation_matrix_to_rpy(R):
    # Convert a rotation matrix to (roll, pitch, yaw) in radians using the ZYX
    # convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy >= 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock (pitch near +/-90 deg): roll and yaw are not separable,
        # so fold everything into roll and report yaw as zero
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    pitch = math.atan2(-R[2, 0], sy)

    return roll, pitch, yaw


def pose_strings(det, K, tag_size):
    # Convert a detected tag pose into a readable translation + roll/pitch/yaw string
    t = det.pose_t.reshape(-1)
    roll, pitch, yaw = (math.degrees(a) for a in rotation_matrix_to_rpy(det.pose_R))

    return (
        f"ID {det.tag_id}: "
        f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}] m, "
        f"rpy=[{roll:.1f},{pitch:.1f},{yaw:.1f}] deg"
    )


def scale_intrinsics(fx, fy, cx, cy, upscale):
    # Preprocessing can resize the image, so the camera intrinsics must be scaled to match
    if None in (fx, fy, cx, cy) or not upscale or upscale == 1.0:
        return fx, fy, cx, cy
    return fx * upscale, fy * upscale, cx * upscale, cy * upscale


def rotation_matrix_to_quaternion(R):
    # Convert a 3x3 rotation matrix into a quaternion
    # This is used for ROS Pose and TF publishing
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    qy = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    qz = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0

    qx = math.copysign(qx, R[2, 1] - R[1, 2])
    qy = math.copysign(qy, R[0, 2] - R[2, 0])
    qz = math.copysign(qz, R[1, 0] - R[0, 1])

    return qx, qy, qz, qw


# Stores recent tag pose history for optional smoothing
_POSE_HISTORY = {}
# Counts consecutive outlier rejections per tag
_POSE_REJECTS = {}


def filter_pose(tag_id, t, args):
    # Smooth tag translations over multiple frames if requested
    window = getattr(args, "pose_window", 1)
    if window <= 1:
        return t

    thresh = getattr(args, "pose_outlier_thresh", 0.0)
    hist = _POSE_HISTORY.get(tag_id, [])

    # Reject obvious outliers if the new pose is too far from recent history
    if len(hist) > 0 and thresh is not None and thresh > 0.0:
        arr_prev = np.stack(hist, axis=0)
        mean_prev = arr_prev.mean(axis=0)
        if np.linalg.norm(t - mean_prev) > thresh:
            # If every recent pose has been rejected, the history itself is probably the
            # outlier (e.g. a bad first detection), so start over from the new pose
            rejects = _POSE_REJECTS.get(tag_id, 0) + 1
            if rejects < window:
                _POSE_REJECTS[tag_id] = rejects
                return None
            hist = []
    _POSE_REJECTS[tag_id] = 0

    # Append current pose and keep only the requested window length
    hist.append(t)
    if len(hist) > window:
        hist = hist[-window:]
    _POSE_HISTORY[tag_id] = hist

    arr = np.stack(hist, axis=0)
    mode = getattr(args, "pose_average", "mean")

    # Return the requested filtered value
    if mode == "median":
        return np.median(arr, axis=0)
    elif mode == "last":
        return hist[-1]
    else:
        return arr.mean(axis=0)
