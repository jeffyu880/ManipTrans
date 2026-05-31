from __future__ import annotations

import argparse
import csv
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


OPTITRACK_TOPIC = "/optitrack/poses"

DEFAULT_OBJECT_FRAME_IDS = ("cap", "bottle")

POSE_FIELDS_PER_FRAME = 7
DEFAULT_OPTITRACK_MAX_AGE_MS = 50.0


@dataclass(frozen=True)
class PoseSample:
    source_wall_s: float
    received_wall_s: float
    pos_x: float
    pos_y: float
    pos_z: float
    ori_x: float
    ori_y: float
    ori_z: float
    ori_w: float

    @classmethod
    def from_msg(cls, msg: PoseStamped, received_wall_s: float) -> "PoseSample":
        source_wall_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if source_wall_s <= 0.0:
            source_wall_s = received_wall_s
        return cls(
            source_wall_s=source_wall_s,
            received_wall_s=received_wall_s,
            pos_x=float(msg.pose.position.x),
            pos_y=float(msg.pose.position.y),
            pos_z=float(msg.pose.position.z),
            ori_x=float(msg.pose.orientation.x),
            ori_y=float(msg.pose.orientation.y),
            ori_z=float(msg.pose.orientation.z),
            ori_w=float(msg.pose.orientation.w),
        )


class OptitrackPoseBuffer(Node):
    def __init__(
        self,
        topic: str = OPTITRACK_TOPIC,
        expected_frame_ids: Sequence[str] = (),
    ) -> None:
        super().__init__("optitrack_data_collection")
        self.expected_frame_ids = tuple(expected_frame_ids)
        self._latest: Dict[str, PoseSample] = {}
        self._lock = threading.Lock()

        self.create_subscription(
            PoseStamped,
            topic,
            self._pose_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Subscribing to {topic}")

    def _pose_callback(self, msg: PoseStamped) -> None:
        frame_id = msg.header.frame_id
        if self.expected_frame_ids and frame_id not in self.expected_frame_ids:
            return
        sample = PoseSample.from_msg(msg, received_wall_s=time.time())
        with self._lock:
            self._latest[frame_id] = sample

    def snapshot(self) -> Dict[str, PoseSample]:
        with self._lock:
            return dict(self._latest)

    def missing_frame_ids(self) -> List[str]:
        with self._lock:
            return [f for f in self.expected_frame_ids if f not in self._latest]

    def wait_for_frames(self, timeout_s: float, poll_interval_s: float = 0.05) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() <= deadline:
            if not self.missing_frame_ids():
                return True
            time.sleep(max(0.0, poll_interval_s))
        return not self.missing_frame_ids()


def pose_csv_header(frame_prefixes: Dict[str, str]) -> List[str]:
    header: List[str] = []
    for prefix in frame_prefixes.values():
        header.extend([
            f"{prefix}_pos_x", f"{prefix}_pos_y", f"{prefix}_pos_z",
            f"{prefix}_ori_x", f"{prefix}_ori_y", f"{prefix}_ori_z", f"{prefix}_ori_w",
        ])
    return header


def pose_csv_row(snapshot: Dict[str, PoseSample], frame_prefixes: Dict[str, str]) -> List[str]:
    row: List[str] = []
    for frame_id in frame_prefixes:
        sample = snapshot.get(frame_id)
        if sample is None:
            row.extend([""] * POSE_FIELDS_PER_FRAME)
            continue
        row.extend([
            f"{sample.pos_x:.9f}", f"{sample.pos_y:.9f}", f"{sample.pos_z:.9f}",
            f"{sample.ori_x:.9f}", f"{sample.ori_y:.9f}", f"{sample.ori_z:.9f}", f"{sample.ori_w:.9f}",
        ])
    return row


def missing_frame_ids_from_snapshot(snapshot: Dict[str, PoseSample], frame_prefixes: Dict[str, str]) -> List[str]:
    return [f for f in frame_prefixes if f not in snapshot]


def sync_csv_header() -> List[str]:
    return ["optitrack_sync_ok", "optitrack_max_age_ms"]


def sync_csv_row(
    snapshot: Dict[str, PoseSample],
    sample_wall_s: float,
    max_allowed_age_ms: float,
    frame_prefixes: Dict[str, str],
) -> List[str]:
    if missing_frame_ids_from_snapshot(snapshot, frame_prefixes):
        return ["0", ""]
    ages_ms = [(sample_wall_s - snapshot[f].source_wall_s) * 1000.0 for f in frame_prefixes]
    max_age_ms = max(ages_ms)
    return [str(int(max_age_ms <= max_allowed_age_ms)), f"{max_age_ms:.3f}"]


def run_collection(
    pose_buffer: OptitrackPoseBuffer,
    log_csv: str,
    frame_prefixes: Dict[str, str],
    optitrack_max_age_ms: float,
    poll_interval_s: float,
) -> None:
    t0 = time.time()
    step = 0

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["step", "timestamp_s"] + pose_csv_header(frame_prefixes) + sync_csv_header()
        writer.writerow(header)

        print(f"Recording to {log_csv} — Ctrl+C to stop")
        while True:
            time.sleep(poll_interval_s)
            snapshot = pose_buffer.snapshot()
            sample_wall_s = time.time()

            row = (
                [step, f"{sample_wall_s - t0:.3f}"]
                + pose_csv_row(snapshot, frame_prefixes)
                + sync_csv_row(snapshot, sample_wall_s, optitrack_max_age_ms, frame_prefixes)
            )
            writer.writerow(row)
            f.flush()

            missing = missing_frame_ids_from_snapshot(snapshot, frame_prefixes)
            sync_ok, max_age = sync_csv_row(snapshot, sample_wall_s, optitrack_max_age_ms, frame_prefixes)
            status = f"sync_ok={sync_ok} max_age_ms={max_age}" if not missing else f"missing={missing}"
            print(f"[step {step}] {status}")
            step += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record OptiTrack object poses to CSV")
    parser.add_argument("--log-csv", default="optitrack_objects_log.csv")
    parser.add_argument("--optitrack-topic", default=OPTITRACK_TOPIC)
    parser.add_argument("--pose-wait-timeout", type=float, default=10.0)
    parser.add_argument("--require-optitrack", action="store_true")
    parser.add_argument("--optitrack-max-age-ms", type=float, default=DEFAULT_OPTITRACK_MAX_AGE_MS)
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Seconds between CSV rows")
    parser.add_argument("--object1-frame", default=DEFAULT_OBJECT_FRAME_IDS[0],
                        help="Motive rigid body name for object 1")
    parser.add_argument("--object2-frame", default=DEFAULT_OBJECT_FRAME_IDS[1],
                        help="Motive rigid body name for object 2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame_prefixes = {
        args.object1_frame: "cap",
        args.object2_frame: "bottle",
    }

    print(f"Tracking frames:")
    for frame_id, prefix in frame_prefixes.items():
        print(f"  {frame_id}  ->  csv prefix: {prefix}")

    rclpy.init()
    pose_buffer = OptitrackPoseBuffer(
        topic=args.optitrack_topic,
        expected_frame_ids=tuple(frame_prefixes.keys()),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(pose_buffer)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if args.pose_wait_timeout > 0.0:
            print(f"Waiting up to {args.pose_wait_timeout:.1f}s for OptiTrack frames...")
            if not pose_buffer.wait_for_frames(args.pose_wait_timeout):
                missing = pose_buffer.missing_frame_ids()
                msg = f"Missing OptiTrack frames: {missing}"
                if args.require_optitrack:
                    raise RuntimeError(msg)
                print(f"[warning] {msg}")

        run_collection(
            pose_buffer=pose_buffer,
            log_csv=args.log_csv,
            frame_prefixes=frame_prefixes,
            optitrack_max_age_ms=args.optitrack_max_age_ms,
            poll_interval_s=args.poll_interval,
        )

    except KeyboardInterrupt:
        print("Stopped by user (Ctrl+C)")
    finally:
        executor.shutdown()
        pose_buffer.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
