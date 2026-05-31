# MotionCapture

ROS2 interface for the OptiTrack Motive motion capture system.
Streams rigid-body poses in real time and converts them to the robot control frame.

## ROS2 package: `optitrack_streamer`

Source in `src/optitrack_streamer/`.  Build with colcon from this directory:

```bash
cd Motion_Capture
colcon build --packages-select optitrack_streamer  --symlink-install
or
colcon build && 

source install/setup.bash

ros2 run optitrack_streamer optitrack_streamer_node 


# ros2 launch optitrack_streamer streamer.launch.py
```

## Coordinate frames

| Frame | Convention | Notes |
|---|---|---|
| Motive global | Y-up | Defined by the calibration square |
| ROS global | Z-up | REP-103 standard |
| Robot base | Z-up | Origin at each arm attachment point |

The node converts Motive quaternions to rotation matrices and republishes poses
in the ROS Z-up frame for direct use by the VMC controller.

## Calibration

```bash
# ros2 launch optitrack_streamer calib_optitrack.launch.py

ros2 run optitrack_streamer optitrack_streamer_node 


## Network:
plug Ethernet cable and set network ip:
    Network settings:
        Address: 169.254.118.70
        Netmask: 255.255.0.0
