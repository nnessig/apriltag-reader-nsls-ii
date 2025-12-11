#!/usr/bin/env python3
# apriltag_runner.py
#
# Detect AprilTags from either:
#   (A) a still image:   --image /path/to/image.png
#   (B) an AVT camera:   --cam-id DEV_xxxxxxxx  (requires VimbaX Python API: vmbpy)
#
# Outputs (when enabled with --save-debug):
#   - <basename>_overlay.png         (input with tag boxes + IDs)
#   - <basename>_edges_overlay.png   (edges for quick visual debugging)
#   - <basename>_gray.png            (preprocessed grayscale frame)
#
# Optional pose if you provide intrinsics + tag size:
#   --tag-size 0.025 --fx ... --fy ... --cx ... --cy ...
#
# Dependencies: pupil_apriltags, opencv-python (cv2), numpy
# Optional (for camera mode): vmbpy (Vimba X)
#
# pip install pupil-apriltags opencv-python numpy
# (Vimba X Python: provided by Allied Vision's SDK)

import os
import sys
import time
import math
import argparse
import numpy as np
import cv2
import queue

# ROS 2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, TransformStamped
import tf2_ros

# Import pupil_apriltags (or apriltag) — using pupil_apriltags for speed/stability
try:
    from pupil_apriltags import Detector
except ImportError as e:
    print("ERROR: pupil_apriltags not found. Install via: pip install pupil-apriltags", file=sys.stderr)
    raise
_DETECTOR_CACHE = {}

# Try importing Vimba X (only needed if using --cam-id)
_vimba_available = False
try:
    import vmbpy as vmb
    _vimba_available = True
except Exception:
    pass

# ----------------------------
# Image preprocessing helpers
# ----------------------------
def apply_clahe(gray, clip_limit=3.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return clahe.apply(gray)

def unsharp_mask(gray, amount=1.0, radius=3):
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def preprocess(img_bgr, use_clahe, use_sharpen, upscale=1.0):
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    if upscale and upscale > 1.0:
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)

    if use_clahe:
        gray = apply_clahe(gray, clip_limit=3.0, grid=(8, 8))

    if use_sharpen:
        gray = unsharp_mask(gray, amount=1.0, radius=1.5)

    return gray

def canny_edges(gray):
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    return edges

# ----------------------------
# Drawing helpers
# ----------------------------
def draw_detections(gray, detections, color=(0, 255, 0)):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in detections:
        pts = d.corners.astype(int)
        for i in range(4):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i + 1) % 4])
            cv2.line(out, pt1, pt2, color, 2)

        center = tuple(d.center.astype(int))
        cv2.circle(out, center, 3, (0, 0, 255), -1)
        cv2.putText(out, f"ID:{d.tag_id}", (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 255), 1, cv2.LINE_AA)
    return out

def draw_edges_overlay(gray, edges):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out[edges > 0] = (0, 255, 255)
    return out

# ----------------------------
# Pose helper (optional)
# ----------------------------
def pose_strings(det, K, tag_size):
    R = det.pose_R
    t = det.pose_t.reshape(-1)
    sy = math.sqrt(R[0, 0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(R[2,1], R[2,2]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        roll = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        yaw = math.degrees(math.atan2(-R[1,2], R[1,1]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        roll = 0.0
    return f"ID {det.tag_id}: t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}] m, rpy=[{roll:.1f},{pitch:.1f},{yaw:.1f}] deg"

# ----------------------------
# AprilTag detection core
# ----------------------------
def detect_tags(gray, families, nthreads, decimate, sigma, refine_edges, decode_sharpening,
                do_pose, tag_size, fx, fy, cx, cy):
    key = (families, nthreads, decimate, sigma, bool(refine_edges), decode_sharpening)
    detector = _DETECTOR_CACHE.get(key)
    if detector is None:
        detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=decimate,
            quad_sigma=sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=False,
        )
        _DETECTOR_CACHE[key] = detector

    if do_pose:
        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size,
        )
    else:
        results = detector.detect(
            gray,
            estimate_tag_pose=False,
        )
    return results

# ----------------------------
# File I/O helpers
# ----------------------------
def write_debug_images(base, gray, edges, overlay, edges_overlay):
    cv2.imwrite(f"{base}_gray.png", gray)
    cv2.imwrite(f"{base}_overlay.png", overlay)
    cv2.imwrite(f"{base}_edges_overlay.png", edges_overlay)
    return [f"{base}_gray.png", f"{base}_overlay.png", f"{base}_edges_overlay.png"]

# ----------------------------
# ROS / math helpers
# ----------------------------
def rotation_matrix_to_quaternion(R):
    qw = math.sqrt(max(0.0, 1.0 + R[0,0] + R[1,0] + R[2,0])) / 2.0
    qx = math.sqrt(max(0.0, 1.0 + R[0,0] - R[1,0] - R[2,0])) / 2.0
    qy = math.sqrt(max(0.0, 1.0 - R[0,0] + R[1,0] - R[2,0])) / 2.0
    qz = math.sqrt(max(0.0, 1.0 - R[0,0] - R[1,0] + R[2,0])) / 2.0
    qx = math.copysign(qx, R[2,1] - R[1,2])
    qy = math.copysign(qy, R[0,2] - R[2,0])
    qz = math.copysign(qz, R[1,0] - R[0,1])
    return qx, qy, qz, qw

_POSE_HISTORY = {}

def filter_pose(tag_id, t, args):
    window = getattr(args, "pose_window", 1)
    if window <= 1:
        return t
    thresh = getattr(args, "pose_outlier_thresh", 0.0)
    hist = _POSE_HISTORY.get(tag_id, [])

    if len(hist) > 0 and thresh is not None and thresh > 0.0:
        arr_prev = np.stack(hist, axis=0)
        mean_prev = arr_prev.mean(axis=0)
        if np.linalg.norm(t - mean_prev) > thresh:
            return None

    hist.append(t)
    if len(hist) > window:
        hist = hist[-window:]
    _POSE_HISTORY[tag_id] = hist

    arr = np.stack(hist, axis=0)
    mode = getattr(args, "pose_average", "mean")
    if mode == "median":
        return np.median(arr, axis=0)
    elif mode == "last":
        return hist[-1]
    else:
        return arr.mean(axis=0)

# ----------------------------
# Camera acquisition (Vimba X)
# ----------------------------
def run_camera(cam_id, args, ros_tag_pub=None, ros_pose_pub=None, ros_node=None, tf_broadcaster=None):
    if not _vimba_available:
        print("ERROR: vmbpy (Vimba X) not available.", file=sys.stderr)
        sys.exit(1)

    import queue
    q = queue.Queue(maxsize=2)

    def handler(cam, stream, frame):
        try:
            if frame.get_status() == vmb.FrameStatus.Complete:
                img = frame.as_numpy_ndarray().copy()
                if q.full():
                    try:
                        _ = q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    q.put_nowait(img)
                except queue.Full:
                    pass
        finally:
            stream.queue_frame(frame)

    with vmb.VmbSystem.get_instance() as vmbs:
        cams = vmbs.get_all_cameras()
        cam = next((c for c in cams if c.get_id() == cam_id), None)
        if cam is None:
            print(f"ERROR: Camera {cam_id} not found. Available: {[c.get_id() for c in cams]}", file=sys.stderr)
            sys.exit(1)

        with cam:
            try:
                try:
                    cam.set_pixel_format(vmb.PixelFormat.Mono8)
                except Exception:
                    pass

                if args.exposure_us is not None:
                    try:
                        cam.ExposureTime.set(float(args.exposure_us))
                    except Exception:
                        pass
                if args.gain is not None:
                    try:
                        cam.Gain.set(float(args.gain))
                    except Exception:
                        pass

                cam.start_streaming(handler=handler, buffer_count=8)

                print("[INFO] Press Ctrl+C to stop.]")
                frame_idx = 0
                last_ros_pub_time = 0.0
                try:
                    while True:
                        try:
                            img = q.get(timeout=2.0)
                        except queue.Empty:
                            print("[WARN] No frame received within timeout.")
                            continue

                        if img.ndim == 3 and img.shape[2] == 1:
                            img = img[:, :, 0]

                        gray = preprocess(img, args.clahe, args.sharpen, args.upscale)

                        do_pose = (args.tag_size is not None and None not in (args.fx, args.fy, args.cx, args.cy))
                        detections = detect_tags(
                            gray,
                            args.families, args.nthreads, args.decimate, args.sigma,
                            args.refine_edges, args.decode_sharpening,
                            do_pose, args.tag_size, args.fx, args.fy, args.cx, args.cy
                        )

                        overlay = draw_detections(gray, detections)

                        base = args.out if args.out else f"frame_{frame_idx:06d}"
                        
                        ids = [int(d.tag_id) for d in detections]

                        publish_ros = (ros_tag_pub is not None) or (ros_pose_pub is not None)
                        now_time = time.time()
                        allow_ros = False
                        if publish_ros:
                            if args.ros_rate is None or args.ros_rate <= 0.0:
                                allow_ros = True
                            else:
                                period = 1.0 / args.ros_rate
                                if now_time - last_ros_pub_time >= period:
                                    allow_ros = True
                                    last_ros_pub_time = now_time

                        if allow_ros and ros_tag_pub is not None:
                            id_msg = Int32MultiArray()
                            id_msg.data = ids
                            ros_tag_pub.publish(id_msg)

                        if allow_ros and ros_pose_pub is not None and do_pose:
                            pose_array = PoseArray()
                            if ros_node is not None:
                                pose_array.header.stamp = ros_node.get_clock().now().to_msg()
                            pose_array.header.frame_id = args.frame_id if hasattr(args, "frame_id") and args.frame_id else "camera"
                            for det in detections:
                                R = det.pose_R
                                t = det.pose_t.reshape(-1)
                                t_f = filter_pose(int(det.tag_id), t, args)
                                if t_f is None:
                                    continue
                                qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                                pose = Pose()
                                pose.position = Point(x=float(t_f[0]), y=float(t_f[1]), z=float(t_f[2]))
                                pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                                pose_array.poses.append(pose)
                                if tf_broadcaster is not None and ros_node is not None:
                                    tf_msg = TransformStamped()
                                    tf_msg.header.stamp = ros_node.get_clock().now().to_msg()
                                    tf_msg.header.frame_id = pose_array.header.frame_id
                                    tf_msg.child_frame_id = f"tag_{int(det.tag_id)}"
                                    tf_msg.transform.translation.x = float(t_f[0])
                                    tf_msg.transform.translation.y = float(t_f[1])
                                    tf_msg.transform.translation.z = float(t_f[2])
                                    tf_msg.transform.rotation.x = qx
                                    tf_msg.transform.rotation.y = qy
                                    tf_msg.transform.rotation.z = qz
                                    tf_msg.transform.rotation.w = qw
                                    tf_broadcaster.sendTransform(tf_msg)
                            ros_pose_pub.publish(pose_array)

                        if frame_idx == 0:
                            run_camera._last_ids = None
                            run_camera._streak = 0
                            run_camera._prev_printed_ids = None

                        if ids and ids == getattr(run_camera, "_last_ids", None):
                            run_camera._streak += 1
                        else:
                            run_camera._streak = 1 if ids else 0
                            run_camera._last_ids = ids

                        interval = getattr(args, "print_every", 30) if getattr(args, "print_every", 0) <= 0 else args.print_every
                        should_print = False
                        if ids:
                            should_print = True
                            if getattr(args, "only_on_change", False):
                                if ids == getattr(run_camera, "_prev_printed_ids", None):
                                    should_print = False
                            if getattr(args, "min_streak", 1) > 1 and run_camera._streak < args.min_streak:
                                should_print = False
                        else:
                            if not getattr(args, "only_on_change", False) and (frame_idx % interval == 0):
                                should_print = True

                        if should_print:
                            if ids:
                                if getattr(args, "save_debug", False) and not getattr(args, "preview", False):
                                    edges = canny_edges(gray)
                                    edges_overlay = draw_edges_overlay(gray, edges)
                                    written = write_debug_images(base, gray, edges, overlay, edges_overlay)
                                    print(f"[{frame_idx:06d}] detections={len(detections)} ids={ids} -> {', '.join(os.path.basename(w) for w in written)}")
                                else:
                                    print(f"[{frame_idx:06d}] detections={len(detections)} ids={ids}")
                                run_camera._prev_printed_ids = ids
                            else:
                                mean_gray = float(np.mean(gray))
                                lapvar = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                                print(f"[{frame_idx:06d}] detections=0 mean={mean_gray:.1f} lapvar={lapvar:.1f}")

                        if getattr(args, "preview", False):
                            cv2.imshow("preview", overlay)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        frame_idx += 1
                        if args.single:
                            break

                except KeyboardInterrupt:
                    print("\n[INFO] Stopping stream.")
                finally:
                    cam.stop_streaming()
                    if getattr(args, "preview", False):
                        try:
                            cv2.destroyAllWindows()
                        except Exception:
                            pass
            except KeyboardInterrupt:
                print("\n[INFO] Stopping stream.")

# ----------------------------
# Image path mode
# ----------------------------
def run_image(image_path, args):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Cannot open {image_path}"
    gray = preprocess(img, args.clahe, args.sharpen, args.upscale)
    edges = canny_edges(gray)
    do_pose = (args.tag_size is not None and None not in (args.fx, args.fy, args.cx, args.cy))
    detections = detect_tags(gray, args.families, args.nthreads, args.decimate, args.sigma,
                             args.refine_edges, args.decode_sharpening,
                             do_pose, args.tag_size, args.fx, args.fy, args.cx, args.cy)
    overlay = draw_detections(gray, detections)
    edges_overlay = draw_edges_overlay(gray, edges)
    if args.out:
        base = args.out
    else:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        base = f"{stem}"
    if getattr(args, "save_debug", False):
        written = write_debug_images(base, gray, edges, overlay, edges_overlay)
        print(f"[INFO] Image detections={len(detections)} -> {', '.join(os.path.basename(w) for w in written)}")
    else:
        print(f"[INFO] Image detections={len(detections)} (no debug images written)")
    if do_pose:
        for d in detections:
            print("   ", pose_strings(d, (args.fx, args.fy, args.cx, args.cy), args.tag_size))

# ----------------------------
# Main / CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="AprilTag detector for image or Vimba X camera")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to input image (e.g., ~/Downloads/test_tag.png)")
    src.add_argument("--cam-id", type=str, help="Vimba camera ID (e.g., DEV_000F315BCAB1)")
    p.add_argument("--families", type=str, default="tag36h11", help="Tag families (comma-separated if multiple)")
    p.add_argument("--decimate", type=float, default=1.0, help="quad_decimate")
    p.add_argument("--sigma", type=float, default=0.0, help="quad_sigma")
    p.add_argument("--nthreads", type=int, default=4, help="detector threads")
    p.add_argument("--refine-edges", action="store_true", help="Enable refine_edges")
    p.add_argument("--decode-sharpening", type=float, default=0.25, help="decode_sharpening (0..1+)")
    p.add_argument("--upscale", type=float, default=1.0, help="Pre-detection upscale factor (e.g., 2.0)")
    p.add_argument("--clahe", action="store_true", help="Apply CLAHE")
    p.add_argument("--sharpen", action="store_true", help="Apply unsharp mask")
    p.add_argument("--exposure-us", type=float, default=None, help="Exposure time (microseconds)")
    p.add_argument("--gain", type=float, default=None, help="Gain (dB)")
    p.add_argument("--tag-size", type=float, default=None, help="Tag size in meters (needed for pose)")
    p.add_argument("--fx", type=float, default=None, help="Camera fx (pixels)")
    p.add_argument("--fy", type=float, default=None, help="Camera fy (pixels)")
    p.add_argument("--cx", type=float, default=None, help="Camera cx (pixels)")
    p.add_argument("--cy", type=float, default=None, help="Camera cy (pixels)")
    p.add_argument("--out", type=str, default=None, help="Base name for output files (no extension)")
    p.add_argument("--single", action="store_true", help="In camera mode, grab a single frame and exit")
    p.add_argument("--no-save", action="store_true", help="Do not write any image files (camera mode, legacy)")
    p.add_argument("--preview", action="store_true", help="Show a live preview window with detections (camera mode)")
    p.add_argument("--ros", action="store_true", help="Publish detected tags to ROS 2")
    p.add_argument("--frame-id", type=str, default="camera", help="Frame ID for published poses")
    p.add_argument("--save-debug", action="store_true", help="Write debug PNG images (gray/overlay/edges)")
    p.add_argument("--ros-rate", type=float, default=20.0, help="Max ROS publish rate in Hz (<=0 for no limit)")
    p.add_argument("--pose-window", type=int, default=1, help="Number of frames to average per tag pose")
    p.add_argument("--pose-outlier-thresh", type=float, default=0.0, help="Outlier threshold in meters (0=off)")
    p.add_argument("--pose-average", type=str, default="mean", help="Pose averaging mode: mean, median, or last")
    return p.parse_args()

def main():
    args = parse_args()

    if args.image:
        run_image(os.path.expanduser(args.image), args)
        return

    if args.ros:
        rclpy.init()
        node = rclpy.create_node("apriltag_runner")
        tag_pub = node.create_publisher(Int32MultiArray, "/apriltag_ids", 10)
        pose_pub = node.create_publisher(PoseArray, "/apriltag_poses", 10)
        tf_broadcaster = tf2_ros.TransformBroadcaster(node)
        try:
            run_camera(args.cam_id, args, ros_tag_pub=tag_pub, ros_pose_pub=pose_pub, ros_node=node, tf_broadcaster=tf_broadcaster)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_camera(args.cam_id, args)

if __name__ == "__main__":
    main()

