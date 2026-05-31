from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare simulation mode and VSA module launch arguments
    calibration_mode_arg = DeclareLaunchArgument(
        'calib',
        default_value='true',
        description='Whether we map the pedals for calibration or not'
    )

    calibration_mode = LaunchConfiguration('calib')

    pedal_node = Node(
        package="input_devices",
        executable="pedal_driver_node",
        name="pedal_driver_node",
        output="screen",
    )

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

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", "/home/kaijunge/.rviz2/default.rviz"],
        output="screen",
    )

    return LaunchDescription([
        pedal_node,
        calibration_mode_arg,
        optitrack_streamer_node,
        wrist_streamer_node,
        rviz_node
     ])
