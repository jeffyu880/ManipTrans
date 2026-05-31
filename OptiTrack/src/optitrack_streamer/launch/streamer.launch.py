from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare simulation mode and VSA module launch arguments
    calibration_mode_arg = DeclareLaunchArgument(
        'calib',
        default_value='false',
        description='Whether we map the pedals for calibration or not'
    )

    calibration_mode = LaunchConfiguration('calib')

    optitrack_streamer_node = Node(
        package="optitrack_streamer",
        executable="optitrack_streamer_node",
        name="optitrack_streamer_node",
        output="screen",
    )
    
    wrist_streamer_node = Node(
        package="optitrack_streamer",
        executable="wrist_streamer_node",
        name="wrist_streamer_node",
        output="screen",
        parameters=[{'calib': calibration_mode}]
    )

    return LaunchDescription([
        calibration_mode_arg,
        optitrack_streamer_node,
        wrist_streamer_node,
     ])
