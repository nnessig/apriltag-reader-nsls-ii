"""ROS 2 publishing of tag IDs, poses, and TF frames.

Importing this module requires ROS 2 (``rclpy``, ``std_msgs``, ``geometry_msgs``,
``tf2_ros``). Import it only when ROS output is requested.
"""

import rclpy
import tf2_ros
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, TransformStamped
from std_msgs.msg import Int32MultiArray

from ..pose import filter_pose, rotation_matrix_to_quaternion


class RosPublisher:
    def __init__(self):
        # Initialize a ROS 2 node and publishers
        rclpy.init()
        self.node = rclpy.create_node("apriltag_runner")
        self.tag_pub = self.node.create_publisher(Int32MultiArray, "/apriltag_ids", 10)
        self.pose_pub = self.node.create_publisher(PoseArray, "/apriltag_poses", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)

    def publish_ids(self, ids):
        # Publish detected tag IDs
        id_msg = Int32MultiArray()
        id_msg.data = ids
        self.tag_pub.publish(id_msg)

    def publish_poses(self, detections, args):
        # Publish detected poses and TF frames
        pose_array = PoseArray()
        pose_array.header.stamp = self.node.get_clock().now().to_msg()

        pose_array.header.frame_id = (
            args.frame_id if hasattr(args, "frame_id") and args.frame_id else "camera"
        )

        for det in detections:
            R = det.pose_R
            t = det.pose_t.reshape(-1)

            # Optionally filter pose over time
            t_f = filter_pose(int(det.tag_id), t, args)
            if t_f is None:
                continue

            qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

            pose = Pose()
            pose.position = Point(x=float(t_f[0]), y=float(t_f[1]), z=float(t_f[2]))
            pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            pose_array.poses.append(pose)

            # Also broadcast a TF frame for each detected tag
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.node.get_clock().now().to_msg()
            tf_msg.header.frame_id = pose_array.header.frame_id
            tf_msg.child_frame_id = f"tag_{int(det.tag_id)}"
            tf_msg.transform.translation.x = float(t_f[0])
            tf_msg.transform.translation.y = float(t_f[1])
            tf_msg.transform.translation.z = float(t_f[2])
            tf_msg.transform.rotation.x = qx
            tf_msg.transform.rotation.y = qy
            tf_msg.transform.rotation.z = qz
            tf_msg.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf_msg)

        self.pose_pub.publish(pose_array)

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()
