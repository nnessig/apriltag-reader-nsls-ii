#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="camera_charuco_calib.yaml",
                    help="Output YAML file")
    ap.add_argument("--show", action="store_true",
                    help="Show detections while running")
    ap.add_argument("--min-markers", type=int, default=2,
                    help="Minimum detected ArUco markers to accept a frame")
    args = ap.parse_args()

    # Rows=3, Cols=5, Checker=5mm, Marker=3.5mm, DICT_4X4
    squares_x = 5
    squares_y = 3
    square_length_m = 0.005  # 5 mm
    marker_length_m = 0.0035  # 3.5 mm
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_m,
        marker_length_m,
        dictionary
    )

    # auto load images from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_paths = sorted(
        glob.glob(os.path.join(script_dir, "*.png")) +
        glob.glob(os.path.join(script_dir, "*.jpg")) +
        glob.glob(os.path.join(script_dir, "*.jpeg"))
    )

    if not img_paths:
        print(f"ERROR: No images found in folder: {script_dir}", file=sys.stderr)
        sys.exit(1)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    kept = 0

    use_new_api = hasattr(cv2.aruco, "ArucoDetector")
    if use_new_api:
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    else:
        detector_params = cv2.aruco.DetectorParameters_create()

    use_charuco_detector = hasattr(cv2.aruco, "CharucoDetector")
    charuco_detector = None
    if use_charuco_detector:
        try:
            charuco_detector = cv2.aruco.CharucoDetector(board)
        except Exception:
            charuco_detector = None

    has_interpolate = hasattr(cv2.aruco, "interpolateCornersCharuco")
    if charuco_detector is None and not has_interpolate:
        raise RuntimeError(
            "cv2.aruco missing CharucoDetector and interpolateCornersCharuco; install an opencv-contrib build."
        )

    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read: {p}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        # detect ArUco markers
        if use_new_api:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)

        if ids is None or len(ids) < args.min_markers:
            if args.show:
                cv2.imshow("charuco", img)
                cv2.waitKey(30)
            continue

        # refine & interpolate Charuco corners
        try:
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=rejected)
        except Exception:
            pass

        if charuco_detector is not None:
            try:
                charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            except Exception:
                charuco_corners, charuco_ids = None, None
        else:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )

        if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
            if args.show:
                vis = img.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imshow("charuco", vis)
                cv2.waitKey(30)
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        kept += 1

        if args.show:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
            cv2.imshow("charuco", vis)
            cv2.waitKey(30)

    if args.show:
        cv2.destroyAllWindows()

    if kept < 7:
        print(f"ERROR: Only kept {kept} good frames. Need ~10+ (prefer 25-40).", file=sys.stderr)
        sys.exit(2)

    # Calibrate
    flags = 0
    K_init = None
    D_init = None

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
        rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=K_init,
            distCoeffs=D_init,
            flags=flags
        )

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    print("\n=== Calibration Result ===")
    print(f"Kept frames: {kept} / {len(img_paths)}")
    print(f"RMS reprojection error: {rms:.4f} pixels")
    print("K (camera matrix):\n", K)
    print("D (dist coeffs):\n", D.reshape(-1))
    print(f"\nUse these in AprilTag runner:")
    print(f"  --fx {fx:.6f} --fy {fy:.6f} --cx {cx:.6f} --cy {cy:.6f}")
    print(f"Image size used: {image_size[0]}x{image_size[1]} (WxH)")

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
