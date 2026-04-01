#!/usr/bin/env python3

import importlib
import sys


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
        return True
    except Exception as exc:
        print(f"[MISSING] {name}: {exc}")
        return False


def main() -> int:
    print("Checking container environment...\n")

    required = [
        "numpy",
        "cv2",
        "pupil_apriltags",
        "rclpy",
        "tf2_ros",
    ]

    optional = [
        "vmbpy",
    ]

    required_ok = True
    for module in required:
        if not check_module(module):
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