"""Billboard text for the Isaac Gym viewer, drawn as world-space line segments.

Isaac Gym's viewer has no 2D text API -- `gymapi.Gym` exposes `add_lines` and body textures and
nothing else -- so an on-screen readout has to be faked: a seven-segment vector font, drawn in the
world on a plane a fixed distance in front of the camera, positioned so it lands in a chosen corner
of the window. Because it is re-anchored to the live camera transform every frame, it tracks the
view as you orbit and stays pinned to the corner.

Deliberately numpy-only (no gymapi), for the same reason as record_cameras.py: it can be imported
from either side without dragging in Isaac Gym or tripping the isaacgym-before-torch import order.
The caller supplies the camera pose as plain arrays and issues the `add_lines` call itself.

Glyph space is a cell 1 wide and 2 tall, origin bottom-left, +y up. Segment names follow the
standard seven-segment layout so the digit table reads like a datasheet:

      A          A = top          D = bottom
    F   B        B = upper right  E = lower left
      G          C = lower right  F = upper left
    E   C        G = middle
      D
"""

from __future__ import annotations

import numpy as np

A = ((0.0, 2.0), (1.0, 2.0))
B = ((1.0, 2.0), (1.0, 1.0))
C = ((1.0, 1.0), (1.0, 0.0))
D = ((0.0, 0.0), (1.0, 0.0))
E = ((0.0, 1.0), (0.0, 0.0))
F = ((0.0, 2.0), (0.0, 1.0))
G = ((0.0, 1.0), (1.0, 1.0))

GLYPHS = {
    "0": (A, B, C, D, E, F),
    "1": (B, C),
    "2": (A, B, G, E, D),
    "3": (A, B, G, C, D),
    "4": (F, G, B, C),
    "5": (A, F, G, C, D),
    "6": (A, F, G, E, C, D),
    "7": (A, B, C),
    "8": (A, B, C, D, E, F, G),
    "9": (A, B, C, D, F, G),
    ".": (((0.35, 0.0), (0.65, 0.0)),),
    "-": (G,),
    "H": (((0.0, 2.0), (0.0, 0.0)), ((1.0, 2.0), (1.0, 0.0)), G),
    "Z": (A, ((1.0, 2.0), (0.0, 0.0)), D),
    # Controller letters for the live readout. D is drawn in its lowercase form on purpose: the
    # uppercase box is identical to "0" on seven segments, and this sits one line under a rate
    # readout full of digits. R and I need strokes the segment set does not have -- R a diagonal
    # leg, I a centre bar -- the same licence "Z" and "H" already take.
    "D": (B, C, D, E, G),
    "R": (F, E, A, B, G, ((0.5, 1.0), (1.0, 0.0))),
    "I": (((0.5, 2.0), (0.5, 0.0)),),
    " ": (),
}

# Per-character pen advance in cell widths. A full cell plus a gap, except the period, which looks
# stranded with a full-width advance after it.
ADVANCE = {".": 0.6, " ": 1.0}
DEFAULT_ADVANCE = 1.4

# Glyph width as a fraction of its height. The cell is 1 x 2 in glyph units, so scaling both axes
# by the same factor would give 0.5 -- noticeably narrow. 0.62 is close to the proportions of a
# real seven-segment display and stops adjacent digits reading as one blob.
WIDTH_RATIO = 0.62

# Straight down the camera's view axis. Far enough not to clip into the near plane, near enough
# that nothing in a table-scale scene (objects sit within ~0.5 m) can occlude it.
DEFAULT_DISTANCE = 1.0

# Fractions of the half-view, insetting the text from the window's top-left corner.
DEFAULT_MARGIN_X = 0.06
DEFAULT_MARGIN_Y = 0.06

# Isaac Gym's default viewer CameraProperties. The viewer is created with a bare
# gymapi.CameraProperties() in vec_task.create_viewer, so this is what it actually has.
DEFAULT_HFOV_DEG = 90.0


def normalize(vector):
    """Unit vector, or the input unchanged if it is degenerate.

    Args:
        vector: (3,) array.

    Returns:
        (3,) float64 unit vector.
    """
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector if norm < 1e-9 else vector / norm


def screen_basis(cam_pos, cam_forward, world_up=(0.0, 0.0, 1.0)):
    """Right/up vectors of the image plane, derived from the view direction alone.

    Built from the world up axis rather than from the camera's own local x/y, so it does not depend
    on which local axis Isaac Gym calls "right" -- only on forward, which its own projectiles.py
    example pins down as local +z.

    right = forward x world_up, NOT world_up x forward. The two differ by a sign and the wrong one
    renders every glyph mirrored. Fix the sign physically: an observer at (0.8, 0, 0.7) facing
    (-1, 0, 0.3) with +z up has their right hand toward +y, and only forward x world_up gives that.
    (In a right-handed frame, facing +x with +z up puts +y on your LEFT -- the ROS x-forward,
    y-left, z-up convention.)

    Args:
        cam_pos: (3,) camera position in world coordinates.
        cam_forward: (3,) camera view direction in world coordinates.
        world_up: (3,) the world's up axis; z-up for this sim.

    Returns:
        (forward, right, up) unit (3,) arrays spanning the view.
    """
    forward = normalize(cam_forward)
    up_axis = normalize(world_up)
    if abs(float(np.dot(forward, up_axis))) > 0.999:
        # looking straight down/up: the cross product is degenerate, so pick any perpendicular
        up_axis = np.array([1.0, 0.0, 0.0])
    right = normalize(np.cross(forward, up_axis))
    up = normalize(np.cross(right, forward))
    return forward, right, up


def corner_anchor(
    cam_pos,
    cam_forward,
    aspect,
    text_height,
    distance=DEFAULT_DISTANCE,
    margin_x=DEFAULT_MARGIN_X,
    margin_y=DEFAULT_MARGIN_Y,
    hfov_deg=DEFAULT_HFOV_DEG,
    world_up=(0.0, 0.0, 1.0),
):
    """World-space origin and axes that place text in the viewer's top-left corner.

    The returned origin is the text's BASELINE-left, already dropped by `text_height` so the
    glyphs hang below the top edge rather than off-screen above it.

    Args:
        cam_pos: (3,) camera position in world coordinates.
        cam_forward: (3,) camera view direction in world coordinates.
        aspect: Viewer width / height.
        text_height: Glyph cell height in world metres at `distance`.
        distance: Metres in front of the camera to place the billboard.
        margin_x: Inset from the left edge, as a fraction of the half-width.
        margin_y: Inset from the top edge, as a fraction of the half-height.
        hfov_deg: Camera horizontal field of view, degrees.
        world_up: (3,) the world's up axis.

    Returns:
        (origin, right, up) -- origin is a (3,) world point, right/up are unit (3,) arrays.
    """
    forward, right, up = screen_basis(cam_pos, cam_forward, world_up)
    half_w = distance * np.tan(np.radians(hfov_deg) / 2.0)
    half_h = half_w / max(aspect, 1e-6)
    centre = np.asarray(cam_pos, dtype=np.float64) + forward * distance
    origin = (
        centre
        - right * (half_w * (1.0 - margin_x))
        + up * (half_h * (1.0 - margin_y) - text_height)
    )
    return origin, right, up


def text_segments(text, origin, right, up, text_height):
    """Lay out `text` as world-space line segments on the plane spanned by right/up.

    Args:
        text: String to draw; characters absent from GLYPHS render as blanks.
        origin: (3,) baseline-left world point.
        right: (3,) unit vector along the text direction.
        up: (3,) unit vector for glyph height.
        text_height: Glyph cell height in world metres.

    Returns:
        (N, 2, 3) float32 endpoint pairs; (0, 2, 3) if nothing was drawable.
    """
    y_scale = text_height / 2.0  # the glyph cell is 2 units tall
    x_scale = text_height * WIDTH_RATIO  # ... and 1 unit wide, widened for legibility
    origin = np.asarray(origin, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    segments = []
    pen = 0.0
    for char in str(text).upper():
        for (x0, y0), (x1, y1) in GLYPHS.get(char, ()):
            segments.append(
                (
                    origin + right * ((pen + x0) * x_scale) + up * (y0 * y_scale),
                    origin + right * ((pen + x1) * x_scale) + up * (y1 * y_scale),
                )
            )
        pen += ADVANCE.get(char, DEFAULT_ADVANCE)
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)
