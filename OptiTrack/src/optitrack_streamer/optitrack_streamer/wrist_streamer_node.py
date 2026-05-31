import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Int16MultiArray
from tf2_ros import TransformBroadcaster
import json, os
import numpy as np
from scipy.spatial.transform import Rotation as R

class WristStreamerNode(Node):

    def __init__(self):
        super().__init__('Wrist_Streamer_Node')
        self.get_logger().info('Starting Wrist Streamer')

        # Parameters
        self.declare_parameter('calib', False)
        self.calib_setting = self.get_parameter('calib').get_parameter_value().bool_value

        # Variables
        self.calib_rotation = {"right":None, "left":None}
        self.calib_translation = {"right":None, "left":None}
        self.state = {"right":"No message", "left":"No message"}
        self.is_pedal = {"right":False, "left":False}
        
        # Static variables
        self.rigid_body_to_wirst = {"right":np.array([0.0, -0.06, -0.12]),"left":np.array([0.0, -0.06, -0.12])}
        self.calib_world_position = {"right":np.array([-0.05, 0.06, 0.12]), "left":np.array([-0.05, 0.06, 0.12])}

        # tf related
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscriber
        self.wrist_raw_subscriber = self.create_subscription(PoseStamped, '/wrist/raw', self.wrist_raw_callback, rclpy.qos.qos_profile_sensor_data)
        self.pedal_subscriber = self.create_subscription(Int16MultiArray, '/pedal', self.pedal_callback, rclpy.qos.qos_profile_sensor_data)

        # Publisher
        self.wrist_publisher = self.create_publisher(PoseStamped, '/wrist/calibrated', rclpy.qos.qos_profile_sensor_data)
           
        # Main loop
        self.mainloop = self.create_timer(1.5, self.mainloop_callback) 

        
    def broadcast_tf(self, base_name, frame_name, position, quaternion):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = base_name
        t.child_frame_id = frame_name

        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_wrist(self, frame_name, position, quaternion):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = frame_name

        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]

        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        self.wrist_publisher.publish(pose)
        
    def get_calibration(self, which_hand):
        dir_path = os.path.dirname(os.path.realpath(__file__))        
        json_file = dir_path + "/../calibration/" + which_hand + ".calib.json"
        try:
            return json.load(open(json_file))
        except:
            return None
        
    def save_calibration(self, which_hand, trans, quat):
        dir_path =  os.path.dirname(os.path.realpath(__file__))  
        json_object = json.dumps({"trans": trans,"quat":quat}, indent=4)
        with open(dir_path + "/../calibration/" + which_hand + ".calib.json", "w") as outfile:
            outfile.write(json_object)

    def pedal_callback(self, msg:Int16MultiArray):
        if msg.data[0] == 1:
            self.is_pedal["left"] = True
        else:
            self.is_pedal["left"] = False

        if msg.data[1] == 1:
            self.is_pedal["right"] = True
        else:
            self.is_pedal["right"] = False

    def wrist_raw_callback(self, msg:PoseStamped):
        for which_hand in ["right", "left"]:
            if msg.header.frame_id == which_hand:

                raw_trans = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
                raw_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, 
                                    msg.pose.orientation.z, msg.pose.orientation.w])
                
                if self.is_pedal[which_hand] and self.calib_setting:
                    self.get_logger().info(which_hand + ": Calibration of glove pose!", throttle_duration_sec=1)
                    self.save_calibration(which_hand, list(raw_trans), list(raw_quat))
                    self.calib = self.get_calibration(which_hand)

                if self.calib_rotation[which_hand] is None:
                    self.calib = self.get_calibration(which_hand)
                    # self.get_logger().info(str(calib))
                    if self.calib is None:
                        self.state[which_hand] = "No calibration"
                        continue
                    
                self.calib_rotation[which_hand] = R.from_quat(self.calib["quat"]).inv()
                self.calib_translation[which_hand] = self.calib["trans"]

                self.state[which_hand] = "Calibrated and streaming"

                # Get the scipy rotation representation of the rigid body aligned with the base coordinate
                aligned = R.from_quat(raw_quat) * self.calib_rotation[which_hand]

                # Get the translation from the 'tree' to the actual wrist of the human
                trans_vec = np.matmul(aligned.as_matrix(), self.rigid_body_to_wirst[which_hand])
                
                # Zero the translation 
                aligned_trans = raw_trans - self.calib_translation[which_hand] + trans_vec + self.calib_world_position[which_hand]
                aligned_quat = aligned.as_quat()

                self.publish_wrist(which_hand, aligned_trans, aligned_quat)
                self.broadcast_tf("world", which_hand + "_ref", aligned_trans, aligned_quat)

    def mainloop_callback(self):
        msg = "\n\n"
        msg += "Right: " + self.state["right"] + "\n"
        msg += "Left: " + self.state["left"] + "\n"
        if self.calib_setting:
            msg += "\nPedal A, B for left, Right hands\n"

        self.get_logger().info(msg)

def main():
    rclpy.init()
    node = WristStreamerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


