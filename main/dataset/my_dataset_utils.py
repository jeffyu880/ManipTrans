"""Per-object physical properties for MyDataset (OptiTrack + AVP) props.

The my_dataset counterpart to `oakink2_dataset_utils.py`'s `oakink2_obj_mass`, kept separate so the
OakInk-v2 table stays exactly as upstream has it. Keyed by the asset id an `ObjectSet` entry
resolves to (see `main/dataset/object_sets.py`), i.e. the name under
`data/my_dataset/obj_files/`, NOT the Motive rigid-body name the capture recorded.

Without an entry here an object keeps whatever `asset_options.density` (200 kg/m^3, low-fill 3D
print) implies from its collision geometry.

Deliberately dependency-free, like `object_sets.py`, so the env can import it without dragging in
torch or tripping the isaacgym import order.
"""

from __future__ import annotations

# asset id -> mass in kg.
#   cup: a receptacle the brush is placed into, not something a hand carries. Its 13 COACD wall
#        pieces come to ~27 cm^3 of convex hull, so density alone gives it only ~5.5 g -- under half
#        the brush's ~12.8 g, light enough that the brush shoves it aside instead of going in.
#        Weighted well past that so it stays put on contact.
my_dataset_obj_mass = {
    "cup": 0.25,
}
