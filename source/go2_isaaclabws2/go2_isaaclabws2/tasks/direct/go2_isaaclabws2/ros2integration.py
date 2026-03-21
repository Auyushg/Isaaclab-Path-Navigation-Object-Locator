import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from cv_bridge import CvBridge
import cv2
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
from isaacsim.ros2.bridge import read_camera_info

class Go2ROS2Bridge(Node):
    def __init__(self, env):
        super().__init__('go2_isaaclab_bridge')
        self.env = env
        self.bridge = CvBridge()
        import threading
        self.lock = threading.Lock()
        self.latest_pos = None
        self.latest_quat = None
        self.latest_vel = None
        self.latest_ang_vel = None

        # Existing publishers
        self.odom_pub = self.create_publisher(Odometry, '/unitree_go2/odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/unitree_go2/pose', 10)
        self.broadcaster = TransformBroadcaster(self)

        # Setup camera publishers via Isaac Sim native pipeline
        self.setup_camera_publishers()

        self.create_timer(0.033, self.publish_state)
        self.camera_info_published = False
        print("[ROS2] Bridge initialized", flush=True)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

    def setup_camera_publishers(self):
        if not hasattr(self.env, 'front_camera'):
            print("[ROS2] No front_camera found", flush=True)
            return

        # Use render_product_paths (plural) and take first element
        render_product = self.env.front_camera._render_product_paths[0]
        print(f"[ROS2] render_product: {render_product}", flush=True)

        # ── RGB ──
        rv_color = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.Rgb.name)
        writer_color = rep.writers.get(rv_color + "ROS2PublishImage")
        writer_color.initialize(
            frameId="unitree_go2/front_cam",
            nodeNamespace="",
            queueSize=1,
            topicName="unitree_go2/front_cam/color_image"
        )
        writer_color.attach([render_product])

        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_color + "IsaacSimulationGate", render_product)
        og.Controller.attribute(gate_path + ".inputs:step").set(1)

        # ── Depth ──
        rv_depth = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name)
        writer_depth = rep.writers.get(rv_depth + "ROS2PublishImage")
        writer_depth.initialize(
            frameId="unitree_go2/front_cam",
            nodeNamespace="",
            queueSize=1,
            topicName="unitree_go2/front_cam/depth_image"
        )
        writer_depth.attach([render_product])

        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_depth + "IsaacSimulationGate", render_product)
        og.Controller.attribute(gate_path + ".inputs:step").set(1)

        # ── Camera Info ──
        camera_info = read_camera_info(render_product_path=render_product)
        writer_info = rep.writers.get("ROS2PublishCameraInfo")
        writer_info.initialize(
            frameId="unitree_go2/front_cam",
            nodeNamespace="",
            queueSize=1,
            topicName="unitree_go2/front_cam/info",
            width=camera_info["width"],
            height=camera_info["height"],
            projectionType=camera_info["projectionType"],
            k=camera_info["k"].reshape([1, 9]),
            r=camera_info["r"].reshape([1, 9]),
            p=camera_info["p"].reshape([1, 12]),
            physicalDistortionModel=camera_info["physicalDistortionModel"],
            physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
        )
        writer_info.attach([render_product])

        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            "PostProcessDispatch" + "IsaacSimulationGate", render_product)
        og.Controller.attribute(gate_path + ".inputs:step").set(1)

        print("[ROS2] Camera publishers set up successfully", flush=True)

    def publish_state(self):
        with self.lock:
            if self.latest_pos is None:
                return
            pos = self.latest_pos.copy()
            quat = self.latest_quat.copy()
            vel = self.latest_vel.copy()
            ang_vel = self.latest_ang_vel.copy()

        stamp = self.get_clock().now().to_msg()

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'unitree_go2/base_link'
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])
        odom.twist.twist.linear.x = float(vel[0])
        odom.twist.twist.linear.y = float(vel[1])
        odom.twist.twist.angular.z = float(ang_vel[2])
        self.odom_pub.publish(odom)

        # Pose
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = 'map'
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)

        # TF: map → base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = "unitree_go2/base_link"
        tf_msg.transform.translation.x = float(pos[0])
        tf_msg.transform.translation.y = float(pos[1])
        tf_msg.transform.translation.z = float(pos[2])
        tf_msg.transform.rotation.w = float(quat[0])
        tf_msg.transform.rotation.x = float(quat[1])
        tf_msg.transform.rotation.y = float(quat[2])
        tf_msg.transform.rotation.z = float(quat[3])
        self.broadcaster.sendTransform(tf_msg)

        
    def _publish_static_transforms(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "unitree_go2/base_link"
        tf_msg.child_frame_id = "unitree_go2/front_cam"

        # Match your TiledCameraCfg offset:
        # pos=(0.4, 0.0, 0.2)
        tf_msg.transform.translation.x = 0.4
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.2

        # Match Zhefan-Xu rotation convention:
        tf_msg.transform.rotation.x = -0.5
        tf_msg.transform.rotation.y = 0.5
        tf_msg.transform.rotation.z = -0.5
        tf_msg.transform.rotation.w = 0.5

        self.static_broadcaster.sendTransform(tf_msg)
        print("[ROS2] Static transform published: base_link → front_cam", flush=True)