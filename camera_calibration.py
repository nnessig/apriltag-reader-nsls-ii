#!/usr/bin/env python3

# camera_calibration.py
#
# Charuco-based camera calibration tool for AprilTag workflows.
#
# This script:
# - Loads calibration images from the script directory
# - Detects ArUco markers and interpolates Charuco corners
# - Computes camera intrinsics (fx, fy, cx, cy) and distortion coefficients
# - Saves results to a YAML file for reuse
#
# Intended for use with apriltag_runner.py for accurate 6-DoF pose estimation.

import argparse
import glob
import os
import sys
import cv2
import numpy as np


def main():
    # Set up command line arguments
    ap = argparse.ArgumentParser()

    # Folder containing calibration images
    ap.add_argument("--images-dir", default="calibration_images",
                    help="Folder containing calibration images")

    # Output yaml file name for saving calibration results
    ap.add_argument("--out", default="output/camera_charuco_calib.yaml",
                    help="Output YAML file")

    # Optionally show detections while the script runs
    ap.add_argument("--show", action="store_true",
                    help="Show detections while running")

    # Minimum number of detected ArUco markers required to keep a frame
    ap.add_argument("--min-markers", type=int, default=2,
                    help="Minimum detected ArUco markers to accept a frame")

    args = ap.parse_args()

    # Board settings (change here if needed for a different setup, original printout generated from calib.io):
    # 5 squares across, 3 squares down
    # Checker square size = 5 mm
    # Marker size = 3.5 mm
    # Using the 4x4 ArUco dictionary with 50 possible markers
    squares_x = 5
    squares_y = 3
    square_length_m = 0.005  # 5 mm
    marker_length_m = 0.0035  # 3.5 mm
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Create the Charuco board object from those settings
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_m,
        marker_length_m,
        dictionary
    )

    # Automatically load all png/jpg/jpeg images from the selected images folder
    images_dir = os.path.abspath(args.images_dir)
    img_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    # Stop immediately if no images were found
    if not img_paths:
        print(f"ERROR: No images found in folder: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # These lists will store all valid detected Charuco corners and ids across every accepted image
    all_charuco_corners = []
    all_charuco_ids = []

    # Will store image size once we read the first valid image
    image_size = None

    # Number of frames that were good enough to keep
    kept = 0

    # Check whether this OpenCV build uses the newer ArucoDetector API
    use_new_api = hasattr(cv2.aruco, "ArucoDetector")
    if use_new_api:
        # New detector setup
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    else:
        # Older detector setup
        detector_params = cv2.aruco.DetectorParameters_create()

    # Check whether this OpenCV build has the newer CharucoDetector API
    use_charuco_detector = hasattr(cv2.aruco, "CharucoDetector")
    charuco_detector = None
    if use_charuco_detector:
        try:
            # Try to create the Charuco detector
            charuco_detector = cv2.aruco.CharucoDetector(board)
        except Exception:
            # If that fails, fall back to older interpolation method
            charuco_detector = None

    # Older OpenCV versions may use interpolateCornersCharuco instead
    has_interpolate = hasattr(cv2.aruco, "interpolateCornersCharuco")

    # If neither CharucoDetector nor interpolateCornersCharuco exists, calibration cannot continue
    if charuco_detector is None and not has_interpolate:
        raise RuntimeError(
            "cv2.aruco missing CharucoDetector and interpolateCornersCharuco; install an opencv-contrib build."
        )

    # Go image by image through the folder
    for p in img_paths:
        # Load image in color
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read: {p}")
            continue

        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save image size from the first successfully read image
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        # Detect ArUco markers in the image
        if use_new_api:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)

        # Skip frame if not enough markers were found
        if ids is None or len(ids) < args.min_markers:
            if args.show:
                cv2.imshow("charuco", img)
                cv2.waitKey(30)
            continue

        # Try to refine the marker detections using the known board layout (this can improve detection accuracy)
        try:
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=rejected)
        except Exception:
            pass

        # Now convert/refine marker detections into Charuco corners
        if charuco_detector is not None:
            try:
                # Newer API directly detects board corners
                charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            except Exception:
                charuco_corners, charuco_ids = None, None
        else:
            # Older API interpolates Charuco corners from marker detections
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )

        # Need at least 4 Charuco corners for a frame to be useful
        if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
            if args.show:
                # show detected markers even if the frame was rejected
                vis = img.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imshow("charuco", vis)
                cv2.waitKey(30)
            continue

        # Keep this frame's Charuco detections for calibration later
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        kept += 1

        # Optionally show the accepted detections
        if args.show:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
            cv2.imshow("charuco", vis)
            cv2.waitKey(30)

    # Clean up any OpenCV windows if visualization was enabled
    if args.show:
        cv2.destroyAllWindows()

    # Require at least some reasonable number of good frames
    # 10 is bare minimum here, but 25-40 is preferred
    if kept < 10:
        print(f"ERROR: Only kept {kept} good frames. Need ~10+ (prefer 25-40).", file=sys.stderr)
        sys.exit(2)

    # Calibration settings
    flags = 0
    K_init = None
    D_init = None

    # Run Charuco calibration
    # Newer OpenCV has an extended version that also returns extra stats
    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        rms, K, D, rvecs, tvecs, stdI, stdE, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=K_init,
            distCoeffs=D_init,
            flags=flags
        )
    else:
        # Older version returns only the main calibration outputs
        rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=K_init,
            distCoeffs=D_init,
            flags=flags
        )

    # Extract intrinsic parameters from camera matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Print summary of calibration result
    print("\n=== Calibration Result ===")
    print(f"Kept frames: {kept} / {len(img_paths)}")
    print(f"RMS reprojection error: {rms:.4f} pixels")
    print("K (camera matrix):\n", K)
    print("D (dist coeffs):\n", D.reshape(-1))

    # Print intrinsics in a format that can be pasted into the AprilTag runner
    print(f"\nUse these in AprilTag runner:")
    print(f"  --fx {fx:.6f} --fy {fy:.6f} --cx {cx:.6f} --cy {cy:.6f}")
    print(f"Image size used: {image_size[0]}x{image_size[1]} (WxH)")

    # Save calibration result to yaml file
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", D)
    fs.write("rms", float(rms))
    fs.release()

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()