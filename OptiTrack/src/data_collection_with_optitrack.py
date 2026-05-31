import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import threading
import time
from .NatNetClient import NatNetClient
from tf2_ros import TransformBroadcaster
from copy import deepcopy as cp



class CustomNatNetClient(NatNetClient):
    def __init__(self, rigid_body_listener=None):
        super().__init__()
        self.rigid_body_listener = rigid_body_listener

class OptitrackStreamerNode(Node):

    def __init__(self):
        super().__init__('Optitrack_streamer')
        self.get_logger().info('Starting Bimanual Optitrack Streamer')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.rigid_bodies = {}
        self.rigid_bodies_lock = threading.Lock()
        self.published_stamp_ns = {}
        self.max_publish_age_s = 0.2

    
        self.pose_publisher = self.create_publisher(PoseStamped, '/optitrack/poses', rclpy.qos.qos_profile_sensor_data)
        
        # ---------------------------------------------------------
        # Mapping of ID of Motive to frame's names
        # Numbers are the IDs of the Rigid Bodies as they appear in Motive, and can be set
        # ---------------------------------------------------------
        self.rb_id_to_frame_id = {
            "80": "Soft_arm_base",
            "81": "Soft_arm_module1",
            "82": "Soft_arm_module2",
            "83": "Soft_arm_module3",
        }

        self.mainloop = self.create_timer(0.01, self.mainloop_callback) 

        self.client = CustomNatNetClient(rigid_body_listener=self.my_rigid_body_listener)
        natnet_thread = threading.Thread(target=self.start_natnet, daemon=True)
        natnet_thread.start()
        
    def start_natnet(self):
        self.client.run()
    
    def my_rigid_body_listener(self, rigid_body_id, position, rotation):
        stamp_ns = time.time_ns()
        with self.rigid_bodies_lock:
            self.rigid_bodies[str(rigid_body_id)] = {
                "pos": position,
                "rot": rotation,
                "stamp_ns": stamp_ns,
            }

    def mainloop_callback(self):
        # Print the IDs of the rigid bodies received from Motive for debugging  
        now_ns = time.time_ns()
        with self.rigid_bodies_lock:
            rigid_body_ids = list(self.rigid_bodies.keys())
            rigid_body_ages_ms = {
                self.rb_id_to_frame_id[rb_id]: round((now_ns - self.rigid_bodies[rb_id]["stamp_ns"]) * 1e-6, 1)
                for rb_id in self.rb_id_to_frame_id
                if rb_id in self.rigid_bodies
            }
        self.get_logger().info(
            "Received RAW IDs: "
            + str(rigid_body_ids)
            + " update_age_ms: "
            + str(rigid_body_ages_ms),
            throttle_duration_sec=2.0,
        )
        
        for id, frame_id in self.rb_id_to_frame_id.items():
            with self.rigid_bodies_lock:
                data = cp(self.rigid_bodies.get(id))
            if data is None:
                continue

            stamp_ns = int(data.get("stamp_ns", time.time_ns()))
            if self.published_stamp_ns.get(id) == stamp_ns:
                continue

            age_s = (now_ns - stamp_ns) * 1e-9
            if age_s > self.max_publish_age_s:
                self.get_logger().warn(
                    f"Skipping stale NatNet pose for {frame_id}: age={age_s:.3f}s",
                    throttle_duration_sec=2.0,
                )
                continue

            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp.sec = stamp_ns // 1_000_000_000
            pose.header.stamp.nanosec = stamp_ns % 1_000_000_000

            # Conversion of RF (Optitrack -> ROS)
            pose.pose.position.x = -data["pos"][0]
            pose.pose.position.y = data["pos"][2]
            pose.pose.position.z = data["pos"][1]

            pose.pose.orientation.x = -data["rot"][0]
            pose.pose.orientation.y = data["rot"][2]
            pose.pose.orientation.z = data["rot"][1]
            pose.pose.orientation.w = data["rot"][3]

            self.pose_publisher.publish(pose)
            self.published_stamp_ns[id] = stamp_ns

def main():
    rclpy.init()
    node = OptitrackStreamerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()