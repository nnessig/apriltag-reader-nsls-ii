#!/usr/bin/env python3

import importlib
import importlib.metadata
import sys


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
        return True
    except Exception as exc:
        print(f"[MISSING] {name}: {exc}")
        return False


def check_opencv_packages() -> bool:
    """opencv-python and opencv-contrib-python share the cv2 package and clash."""
    found = [
        name
        for name in ("opencv-python", "opencv-python-headless", "opencv-contrib-python")
        if _installed(name)
    ]
    if len(found) > 1:
        print(f"[CONFLICT] multiple OpenCV pip packages installed: {', '.join(found)}")
        return False
    print(f"[OK] OpenCV pip package: {found[0] if found else 'none'}")
    return True


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def main() -> int:
    print("Checking container environment...\n")

    required = [
        "numpy",
        "cv2",
        "cv2.aruco",
        "pupil_apriltags",
        "yaml",
        "setuptools",
        "rclpy",
        "tf2_ros",
        "std_msgs.msg",
        "geometry_msgs.msg",
    ]

    optional = [
        "vmbpy",
    ]

    required_ok = True
    for module in required:
        if not check_module(module):
            required_ok = False

    if not check_opencv_packages():
        required_ok = False

    print("\nOptional:")
    for module in optional:
        check_module(module)

    print("\nSummary:")
    if required_ok:
        print("Container has the required Python + ROS modules.")
    else:
        print("Container is missing one or more required modules.")
        return 1

    print("If vmbpy is missing, mount Vimba X into /opt/VimbaX and restart the container.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
