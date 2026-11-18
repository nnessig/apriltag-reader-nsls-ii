"""Command-line entry point: image mode, camera mode, and optional ROS 2 output.

AprilTag detection tool for either:
- A still image, or
- A live Allied Vision (AVT) camera using the Vimba X SDK

It can detect AprilTags, apply optional preprocessing, estimate 6-DoF pose when
camera intrinsics and tag size are given, save debug images, and publish IDs,
poses, and TF frames to ROS 2.

ROS 2 and Vimba X are optional and are only imported when requested.
"""

import argparse
import os
import queue
import sys
import time

import cv2
import numpy as np

from .detector import detect_tags
from .pose import pose_strings, scale_intrinsics
from .preprocessing import canny_edges, preprocess
from .rendering import draw_detections, draw_edges_overlay, write_debug_images


def run_camera(cam_id, args, ros=None):
    # Imported here so image mode never needs the camera backend
    from .cameras.vimba import open_stream

    try:
        with open_stream(cam_id, args.exposure_us, args.gain) as q:
            print("[INFO] Press Ctrl+C to stop.]")
            frame_idx = 0
            last_ros_pub_time = 0.0
            last_ids = None
            streak = 0
            prev_printed_ids = None

            try:
                while True:
                    # Wait for the latest frame
                    try:
                        img = q.get(timeout=2.0)
                    except queue.Empty:
                        print("[WARN] No frame received within timeout.")
                        continue

                    # Flatten single-channel 3D arrays into 2D if needed
                    if img.ndim == 3 and img.shape[2] == 1:
                        img = img[:, :, 0]

                    # Preprocess image before tag detection
                    gray = preprocess(img, args.clahe, args.sharpen, args.upscale)

                    # Pose estimation only works if tag size and camera intrinsics are all provided
                    do_pose = args.tag_size is not None and None not in (
                        args.fx,
                        args.fy,
                        args.cx,
                        args.cy,
                    )

                    # Run AprilTag detection
                    fx, fy, cx, cy = scale_intrinsics(
                        args.fx, args.fy, args.cx, args.cy, args.upscale
                    )
                    detections = detect_tags(
                        gray,
                        args.families,
                        args.nthreads,
                        args.decimate,
                        args.sigma,
                        args.refine_edges,
                        args.decode_sharpening,
                        do_pose,
                        args.tag_size,
                        fx,
                        fy,
                        cx,
                        cy,
                    )

                    # Draw detection results for optional preview/debug
                    overlay = draw_detections(gray, detections)

                    # Build output filename base
                    base = args.out if args.out else f"frame_{frame_idx:06d}"

                    # Extract integer tag IDs
                    ids = [int(d.tag_id) for d in detections]

                    # Decide whether it is time to publish ROS output
                    now_time = time.time()
                    allow_ros = False

                    if ros is not None:
                        if args.ros_rate is None or args.ros_rate <= 0.0:
                            allow_ros = True
                        else:
                            period = 1.0 / args.ros_rate
                            if now_time - last_ros_pub_time >= period:
                                allow_ros = True
                                last_ros_pub_time = now_time

                    if allow_ros:
                        ros.publish_ids(ids)
                        if do_pose:
                            ros.publish_poses(detections, args)

                    # Track whether the same IDs are being seen repeatedly
                    if ids and ids == last_ids:
                        streak += 1
                    else:
                        streak = 1 if ids else 0
                        last_ids = ids

                    # Control console print frequency
                    interval = (
                        getattr(args, "print_every", 30)
                        if getattr(args, "print_every", 0) <= 0
                        else args.print_every
                    )

                    should_print = False

                    if ids:
                        should_print = True

                        if getattr(args, "only_on_change", False):
                            if ids == prev_printed_ids:
                                should_print = False

                        if getattr(args, "min_streak", 1) > 1 and streak < args.min_streak:
                            should_print = False
                    else:
                        if not getattr(args, "only_on_change", False) and (
                            frame_idx % interval == 0
                        ):
                            should_print = True

                    # Print detection summary and optionally save debug images
                    if should_print:
                        if ids:
                            if getattr(args, "save_debug", False) and not getattr(
                                args, "preview", False
                            ):
                                edges = canny_edges(gray)
                                edges_overlay = draw_edges_overlay(gray, edges)
                                written = write_debug_images(
                                    base, gray, edges, overlay, edges_overlay
                                )
                                print(
                                    f"[{frame_idx:06d}] detections={len(detections)} "
                                    f"ids={ids} -> {', '.join(os.path.basename(w) for w in written)}"
                                )
                            else:
                                print(f"[{frame_idx:06d}] detections={len(detections)} ids={ids}")

                            prev_printed_ids = ids
                        else:
                            # Print quick image stats if no tags were found
                            mean_gray = float(np.mean(gray))
                            lapvar = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                            print(
                                f"[{frame_idx:06d}] detections=0 mean={mean_gray:.1f} lapvar={lapvar:.1f}"
                            )

                    # Show live preview if requested
                    if getattr(args, "preview", False):
                        cv2.imshow("preview", overlay)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    frame_idx += 1

                    # Single-frame mode exits after one processed frame
                    if args.single:
                        break

            finally:
                if getattr(args, "preview", False):
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\n[INFO] Stopping stream.")


def run_image(image_path, args):
    # Load a single image from disk
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: Cannot open image: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Preprocess the image
    gray = preprocess(img, args.clahe, args.sharpen, args.upscale)
    edges = canny_edges(gray)

    # Pose estimation is only enabled if full camera intrinsics and tag size are given
    do_pose = args.tag_size is not None and None not in (args.fx, args.fy, args.cx, args.cy)

    # Run AprilTag detection on the image
    fx, fy, cx, cy = scale_intrinsics(args.fx, args.fy, args.cx, args.cy, args.upscale)
    detections = detect_tags(
        gray,
        args.families,
        args.nthreads,
        args.decimate,
        args.sigma,
        args.refine_edges,
        args.decode_sharpening,
        do_pose,
        args.tag_size,
        fx,
        fy,
        cx,
        cy,
    )

    # Build visualization images
    overlay = draw_detections(gray, detections)
    edges_overlay = draw_edges_overlay(gray, edges)

    # Choose output basename
    if args.out:
        base = args.out
    else:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        base = f"{stem}"

    # Save debug images if requested
    if getattr(args, "save_debug", False):
        written = write_debug_images(base, gray, edges, overlay, edges_overlay)
        print(
            f"[INFO] Image detections={len(detections)} -> {', '.join(os.path.basename(w) for w in written)}"
        )
    else:
        print(f"[INFO] Image detections={len(detections)} (no debug images written)")

    # Print pose summaries if pose estimation was enabled
    if do_pose:
        for d in detections:
            print("   ", pose_strings(d, (args.fx, args.fy, args.cx, args.cy), args.tag_size))


def parse_args():
    # Set up command line arguments
    p = argparse.ArgumentParser(description="AprilTag detector for image or Vimba X camera")

    # User must choose exactly one input source: image or live camera
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image", type=str, help="Path to input image (e.g., ~/Downloads/test_tag.png)"
    )
    src.add_argument("--cam-id", type=str, help="Vimba camera ID (e.g., DEV_000F315BCAB1)")

    # Detector configuration
    p.add_argument(
        "--families",
        type=str,
        default="tag36h11",
        help="Tag families (comma-separated if multiple)",
    )
    p.add_argument("--decimate", type=float, default=1.0, help="quad_decimate")
    p.add_argument("--sigma", type=float, default=0.0, help="quad_sigma")
    p.add_argument("--nthreads", type=int, default=4, help="detector threads")
    p.add_argument("--refine-edges", action="store_true", help="Enable refine_edges")
    p.add_argument(
        "--decode-sharpening", type=float, default=0.25, help="decode_sharpening (0..1+)"
    )

    # Image preprocessing options
    p.add_argument(
        "--upscale", type=float, default=1.0, help="Pre-detection upscale factor (e.g., 2.0)"
    )
    p.add_argument("--clahe", action="store_true", help="Apply CLAHE")
    p.add_argument("--sharpen", action="store_true", help="Apply unsharp mask")

    # Camera exposure / gain options
    p.add_argument("--exposure-us", type=float, default=None, help="Exposure time (microseconds)")
    p.add_argument("--gain", type=float, default=None, help="Gain (dB)")

    # Pose estimation inputs
    p.add_argument(
        "--tag-size", type=float, default=None, help="Tag size in meters (needed for pose)"
    )
    p.add_argument("--fx", type=float, default=None, help="Camera fx (pixels)")
    p.add_argument("--fy", type=float, default=None, help="Camera fy (pixels)")
    p.add_argument("--cx", type=float, default=None, help="Camera cx (pixels)")
    p.add_argument("--cy", type=float, default=None, help="Camera cy (pixels)")

    # Output and run behavior
    p.add_argument(
        "--out", type=str, default=None, help="Base name for output files (no extension)"
    )
    p.add_argument(
        "--single", action="store_true", help="In camera mode, grab a single frame and exit"
    )
    p.add_argument(
        "--no-save", action="store_true", help="Do not write any image files (camera mode, legacy)"
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window with detections (camera mode)",
    )
    p.add_argument(
        "--save-debug", action="store_true", help="Write debug PNG images (gray/overlay/edges)"
    )

    # ROS output options
    p.add_argument("--ros", action="store_true", help="Publish detected tags to ROS 2")
    p.add_argument("--frame-id", type=str, default="camera", help="Frame ID for published poses")
    p.add_argument(
        "--ros-rate", type=float, default=20.0, help="Max ROS publish rate in Hz (<=0 for no limit)"
    )

    # Optional pose filtering settings
    p.add_argument(
        "--pose-window", type=int, default=1, help="Number of frames to average per tag pose"
    )
    p.add_argument(
        "--pose-outlier-thresh", type=float, default=0.0, help="Outlier threshold in meters (0=off)"
    )
    p.add_argument(
        "--pose-average",
        type=str,
        default="mean",
        help="Pose averaging mode: mean, median, or last",
    )

    return p.parse_args()


def main():
    # Parse user arguments
    args = parse_args()

    # If the source is a still image, process it and exit
    if args.image:
        run_image(os.path.expanduser(args.image), args)
        return

    # If ROS output is enabled, initialize a ROS 2 node and publishers
    if args.ros:
        try:
            from .ros.publisher import RosPublisher
        except ImportError as e:
            print(f"ERROR: ROS 2 Python packages not available: {e}", file=sys.stderr)
            sys.exit(1)

        ros = RosPublisher()
        try:
            run_camera(args.cam_id, args, ros=ros)
        finally:
            ros.shutdown()

    # Otherwise just run the live camera pipeline without ROS
    else:
        run_camera(args.cam_id, args)


if __name__ == "__main__":
    main()
