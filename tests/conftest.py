"""Shared test setup.

apriltag_runner.py imports the ROS 2 Python packages at module level. When ROS 2
is not installed (for example on a plain CI runner or a laptop), substitute
mock modules so the pure-Python helpers can still be imported and tested. When
ROS 2 is installed, the real packages are used.
"""

import sys
from unittest import mock

try:
    import rclpy  # noqa: F401
except ImportError:
    for _name in (
        "rclpy",
        "rclpy.node",
        "std_msgs",
        "std_msgs.msg",
        "geometry_msgs",
        "geometry_msgs.msg",
        "tf2_ros",
    ):
        sys.modules[_name] = mock.MagicMock()
