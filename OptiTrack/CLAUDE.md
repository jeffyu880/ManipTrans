# OptiTrack Data Collection — CLAUDE.md

## System Overview

This repo collects synchronized motor and pose data from a 3-module soft robotic arm.
It drives 9 Dynamixel motors through random step sequences and records each motor's
commanded and readback positions alongside 6-DOF rigid-body poses from an OptiTrack
motion-capture system, all written to a single CSV per run.

**Two processes must run simultaneously:**
1. `optitrack_streamer_node` — ROS2 node bridging NatNet (Motive) → `/optitrack/poses`
2. `data_collection_with_optitrack.py` — main script commanding motors and logging

---

## Hardware

| Component | Detail |
|---|---|
| Dynamixel motors | 9 motors, IDs 1–9; 3 per arm module |
| Serial bus | USB-to-serial, default `/dev/ttyUSB0`, 1 Mbaud |
| OptiTrack markers | 4 rigid bodies defined in Motive |
| OptiTrack PC | Windows machine running Motive, IP `169.254.118.69` |
| Linux PC (this machine) | IP `169.254.118.70`, netmask `255.255.0.0` (link-local ethernet) |

### Motor → Module → OptiTrack mapping

| Motors | Module frame | Motive rigid body ID |
|---|---|---|
| 1–3 | `Soft_arm_module1` | 81 |
| 4–6 | `Soft_arm_module2` | 82 |
| 7–9 | `Soft_arm_module3` | 83 |
| (base) | `Soft_arm_base` | 80 |

---

## Repository Layout

```
OptiTrack/
├── src/
│   ├── data_collection_with_optitrack.py   # main collection script
│   └── optitrack_streamer/                 # ROS2 package
│       ├── optitrack_streamer/
│       │   ├── optitrack_streamer_node.py  # NatNet → /optitrack/poses
│       │   ├── NatNetClient.py             # NatNet UDP protocol (NaturalPoint)
│       │   ├── MoCapData.py
│       │   └── DataDescriptions.py
│       ├── launch/streamer.launch.py
│       └── package.xml
├── encoder_log.csv       # INPUT: motor zero positions (last row used)
├── data_collection_optitrack_log_fixed.csv  # example output
├── install/              # colcon install tree (source install/setup.bash)
└── build/
```

The `data_collection` Python module (imported as `from data_collection import ...`) lives
**outside this repo** and must be on `PYTHONPATH`. It provides `DynamixelController`,
`DEFAULT_ACTION_RANGES`, `build_relative_command`, `sequence_assignment`,
`apply_relative_slew_limit`, `set_motion_profile_all`, and `MAX_PROFILE_VALUE`.

---

## Dependencies

### System / ROS2
- ROS2 Humble (Python, ament_python build type)
- `rclpy`, `geometry_msgs`, `tf2_ros` (standard ROS2 packages)
- Python 3.10 (matches installed `.pyc` files)

### Python packages
- `numpy`
- `dynamixel_sdk` (used by `data_collection` module)
- `data_collection` module — must be importable from `PYTHONPATH`

### Network
- Ethernet link-local between Linux PC and Motive PC
- NatNet unicast (`use_multicast = False` in `NatNetClient.py:80`)
- Motive streaming must be enabled; rigid body IDs **must** be 80, 81, 82, 83

---

## Setup Checklist (do before every run)

1. **Network interface** — set Linux NIC to static IP `169.254.118.70 / 255.255.0.0`
   ```
   Network settings → Address: 169.254.118.70, Netmask: 255.255.0.0
   ```

2. **Motive (Windows PC)**
   - Motive must be running and streaming over NatNet
   - Rigid bodies must be assigned IDs **80, 81, 82, 83** (not names—numeric IDs in Motive)
   - Streaming IP must be `169.254.118.69`

3. **Build the ROS2 package** (only needed once, or after source changes):
   ```bash
   cd /home/jeffrey/Documents/Manipulation/OptiTrack
   colcon build --packages-select optitrack_streamer --symlink-install
   ```

4. **Source ROS2 and the workspace** (every new terminal):
   ```bash
   source /opt/ros/humble/setup.bash
   source /home/jeffrey/Documents/Manipulation/OptiTrack/install/setup.bash
   ```

5. **Ensure `data_collection` module is importable** — add its directory to `PYTHONPATH`:
   ```bash
   export PYTHONPATH=/path/to/data_collection_module:$PYTHONPATH
   ```

6. **Zero-pose CSV** — `encoder_log.csv` must exist and have at least one data row.
   Columns: `timestamp_s, id_1, id_2, ..., id_9` (raw encoder counts; last row is used).

7. **Dynamixel bus** — USB-to-serial adapter connected, device visible at `/dev/ttyUSB0`.
   Check with `ls /dev/ttyUSB*`. May need `sudo chmod 666 /dev/ttyUSB0` or add user to `dialout`.

---

## Running

### Terminal 1 — OptiTrack streamer
```bash
source /opt/ros/humble/setup.bash
source /home/jeffrey/Documents/Manipulation/OptiTrack/install/setup.bash
ros2 run optitrack_streamer optitrack_streamer_node
```
Verify you see log lines like:
```
Received RAW IDs: ['80', '81', '82', '83'] update_age_ms: {Soft_arm_base: 5.2, ...}
```

### Terminal 2 — Data collection
```bash
source /opt/ros/humble/setup.bash
source /home/jeffrey/Documents/Manipulation/OptiTrack/install/setup.bash
cd /home/jeffrey/Documents/Manipulation/OptiTrack
python3 src/data_collection_with_optitrack.py \
    --port /dev/ttyUSB0 \
    --zero-csv encoder_log.csv \
    --log-csv data_collection_optitrack_log.csv \
    --episodes 1 \
    --settle 1.0 \
    --seed 42
```

Ctrl+C stops collection early. The arm always attempts to rest back to zero on exit.

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--port` | `/dev/ttyUSB0` | Serial device for Dynamixel bus |
| `--baud` | `1000000` | Baudrate |
| `--zero-csv` | `encoder_log.csv` | CSV whose last row gives motor zero positions |
| `--log-csv` | `data_collection_optitrack_log.csv` | Output CSV path |
| `--period-steps` | `9000` | Steps per episode (one full motion period T) |
| `--settle` | `1.0` | Seconds to wait after each command before readback |
| `--seed` | `42` | NumPy RNG seed |
| `--episodes` | `1` | Max episodes to collect; `0` = run until Ctrl+C |
| `--max-rel-step` | `0.12` | Max normalized command change per step; `0` disables |
| `--profile-vel` | `30` | Dynamixel Profile Velocity (0–32767; lower = slower) |
| `--profile-acc` | `5` | Dynamixel Profile Acceleration (0–32767; lower = softer) |
| `--skip-rest-to-zero` | off | Skip initial homing move to zero pose |
| `--require-optitrack` | off | Abort if OptiTrack frames not received before collection |
| `--pose-wait-timeout` | `10.0` | Seconds to wait for all 4 OptiTrack frames at startup |
| `--optitrack-max-age-ms` | `50.0` | Pose age threshold for `optitrack_sync_ok=1` flag |
| `--optitrack-topic` | `/optitrack/poses` | ROS2 topic for PoseStamped messages |

---

## Inputs

| Input | Path | Format |
|---|---|---|
| Motor zero positions | `encoder_log.csv` | CSV with header `timestamp_s,id_1,...,id_9`; last row used |
| Arm poses | ROS2 topic `/optitrack/poses` | `geometry_msgs/PoseStamped`; `frame_id` ∈ {`Soft_arm_base`, `Soft_arm_module1`, `Soft_arm_module2`, `Soft_arm_module3`} |

---

## Output CSV Schema

Output file (default `data_collection_optitrack_log.csv`):

| Column(s) | Description |
|---|---|
| `step` | Global step counter |
| `timestamp_s` | Wall-clock seconds since collection start |
| `episode` | Episode index (`step // period_steps`) |
| `motor1_cmd` … `motor9_cmd` | Raw commanded encoder counts |
| `motor1_read` … `motor9_read` | Raw readback encoder counts |
| `base_pos_x/y/z`, `base_ori_x/y/z/w` | Base rigid body pose (ROS frame, metres + quaternion) |
| `module1_pos_x/y/z`, `module1_ori_x/y/z/w` | Module 1 pose |
| `module2_pos_x/y/z`, `module2_ori_x/y/z/w` | Module 2 pose |
| `module3_pos_x/y/z`, `module3_ori_x/y/z/w` | Module 3 pose |
| `optitrack_sync_ok` | `1` if all 4 frames present and max pose age ≤ `optitrack_max_age_ms` |
| `optitrack_max_age_ms` | Age of the stalest pose at row timestamp (ms) |

Pose cells are empty strings if a frame was not yet received.

---

## Coordinate Frame Conversion

OptiTrack Motive uses Y-up; ROS uses Z-up (REP-103). The streamer node converts:

```
ROS position:    x = -Motive_x,  y = Motive_z,  z = Motive_y
ROS quaternion:  x = -Motive_qx, y = Motive_qz, z = Motive_qy, w = Motive_qw
```

---

## NatNet Network Settings (hardcoded in NatNetClient.py)

| Parameter | Value |
|---|---|
| Server IP (Motive PC) | `169.254.118.69` |
| Client IP (Linux PC) | `169.254.118.70` |
| Multicast address | `239.255.42.99` (unused; unicast mode) |
| `use_multicast` | `False` |

To change IPs, edit `src/optitrack_streamer/optitrack_streamer/NatNetClient.py` lines 66–69.
