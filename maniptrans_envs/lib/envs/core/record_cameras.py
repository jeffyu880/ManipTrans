"""Camera poses for off-screen recording, shared by the env and the playback script.

Both `dexhandmanip_bih.create_camera`/`create_camera_top` (what `capture_video=true` records
through) and `data_stats/playback_trajectory.py --record` render the same table from the same
places, so the two sets of footage are visually comparable. They lived as separate literals and
had already drifted — playback sat on an oblique 3/4 view while the env used a head-on front and a
behind view — which made a policy recording and a retarget playback of the same demo impossible to
put side by side.

Deliberately stdlib-only (plain tuples, no gymapi) so either side can import it without dragging
in Isaac Gym; callers wrap the tuples in `gymapi.Vec3` themselves.

Poses are (eye, target) in the gym world frame, metres. The table surface sits at z = 0.415 and
its centre near x = -0.1, so both views look slightly down onto the working area.
"""

# Head-on, from the +x side looking back across the table. The primary recording view.
FRONT_EYE = (0.80, 0.0, 0.7)
FRONT_TARGET = (-1.0, 0.0, 0.3)

# From behind the hands, looking the other way. Saved as the `_top` video by the env's recorder,
# despite the name -- it is a behind view, not an overhead one.
BEHIND_EYE = (-0.97, 0.0, 0.74)
BEHIND_TARGET = (1.0, 0.0, 0.3)

# Matches a typical webcam; kept identical across both views so the two are directly comparable.
RECORD_FOV = 69.4
RECORD_WIDTH = 1280
RECORD_HEIGHT = 720

VIEWS = {
    "front": (FRONT_EYE, FRONT_TARGET),
    "behind": (BEHIND_EYE, BEHIND_TARGET),
}
