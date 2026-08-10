---
name: coding-rules
description: Code style and conventions for the ManipTrans repo. Read before writing or editing any Python, YAML config, or SLURM script here — including the non-negotiable Python function rules (no leading underscores, docstring on every function, one-line plain-English gloss).
---

# ManipTrans coding rules

Two kinds of rule live here. **Python function style** below is a personal house style, applied
automatically to anything Claude writes or reviews, whether or not style was mentioned. Everything
after it is conventions inferred from the existing code — there is no formatter or linter
configured, so they are enforced by reading.

## Python function style (non-negotiable)

Applies whenever writing, reviewing, or refactoring Python — especially research/robotics/ML code
(training scripts, env wrappers, data pipelines).

1. **No leading underscores on function names.**
   Even "internal" helpers get plain names: `build_env`, not `_build_env`. Leading underscores are
   a weak, inconsistent privacy convention in Python and add visual noise; signal "internal" with
   clear naming and module boundaries instead. Dunder methods (`__init__`, `__repr__`, …) are
   exempt — the language requires them.

2. **Every function has a docstring.**
   Triple-quoted, directly under the `def`. Minimum: a one-line summary, `Args:`, and `Returns:`
   (drop `Returns:` only when the function genuinely returns nothing). For tensor/array-heavy code
   — most of this repo — put shapes in the docstring: `pos: (B, T, 3) world-frame positions`.

3. **Every function has a brief plain-English explanation.**
   Distinct from the docstring body: a short gloss, either as the docstring's first line or a
   one-line comment directly above the `def`, so intent is legible without reading the whole thing.
   One short sentence. Don't restate the name — say what it does or why it exists.

```python
def compute_grasp_success(contact_forces, threshold=1.0):
    """Check whether contact forces indicate a stable grasp.

    Args:
        contact_forces: (B, N_fingers) contact force magnitudes per finger.
        threshold: Minimum force (N) required per finger to count as contact.

    Returns:
        (B,) bool tensor, True where all fingers meet the threshold.
    """
    return (contact_forces > threshold).all(dim=-1)
```

Flag and fix anything like this — underscore prefix, no docstring, no explanation:

```python
def _load_demo(path):
    return np.load(path)
```

**Applying it.** Writing new code: all three rules by default, unasked, for every function.
Reviewing or refactoring: call out every function that has a leading underscore, lacks a docstring,
or lacks a gloss — fix them directly if a refactor was requested, list them if it was a review.
Don't over-apply: this governs named `def`s, not one-line lambdas or trivial script-level code. Any
named `def` gets the full treatment. Broader structural concerns (function length, config hoisting,
frame-aware tensor naming, guard clauses, logging vs. print, import ordering, god files) are worth
raising in review, but these three are the strict baseline.

**Note on this repo.** The existing code contradicts rule 1 heavily — `_create_prop_actor`,
`_load_obj_asset`, `_object_reward_shares` and most other env methods are underscore-prefixed, and
many functions have no docstring. Treat the rule as going-forward: apply it to new functions, and
to functions you are already rewriting. Do not mass-rename existing methods, which would break
call sites across the env, the loaders and the network builders for no functional gain.

## Comments

The repo's defining habit: **comments explain why, never what**. Density runs 12–19% of non-blank
lines. A comment that restates the code is noise; a comment that records the reason a value or
branch exists is the point.

Every magic number carries its derivation:

```python
#   bottle_body: origin is 4.88 cm above its base -> +0.05 leaves the base 0.12 cm above the table.
#   cup:         origin IS its base              -> +0.05 would float the whole scene 5 cm.
RECENTER_FINE = (0.0, 0.05, 0.0)
```

Not `RECENTER_FINE = (0.0, 0.05, 0.0)  # fine recentering offset`.

Cross-reference by path or symbol so the reader can follow the thread: `see
main/dataset/object_sets.py`, `matches the env, which collapses the two scored actors`. When two
places must agree, say so at both ends — or better, delete one of them (below).

Module docstrings state what the module owns and how it fits the system, not just its contents.
`main/dataset/object_sets.py` is the model: purpose, the two call paths it serves, why it lives
apart, and how to extend it.

## Line length

~100 characters. p95 across the codebase sits at 98–102. Occasional long lines exist; don't add
more.

## Single ownership

If a constant or rule is needed in two modules, it gets one owner and both import it. The repo
already paid for the alternative — `OBJ_ASSETS` and `RECENTER_FINE` were duplicated across the two
loaders behind `!!! KEEP IDENTICAL !!!` banners until they were pulled into `object_sets.py`. Don't
reintroduce that pattern; a shared module is always the answer.

## Assertions carry the fix

Failures name the offending value and say what to do about it:

```python
assert not missing, (
    f"obj_id '{obj_id}' is not in OBJ_ASSETS and these files are missing: {missing}. "
    f"Either add it to OBJ_ASSETS or name the files after the capture's rigid body."
)
```

Prefer an explicit assert over letting a typo surface three frames later as a `KeyError`, and guard
sentinel returns — `find_actor_handle` gives `-1` for a missing actor, which silently indexes the
last one.

## New behaviour defaults to old behaviour

A new flag, table entry, or config knob must be a no-op until switched on, and say so in its
comment. Precedents: `recenter_fine` is per-object-set with bottle unchanged; `my_dataset_obj_mass`
leaves any object without an entry on its density-derived mass; `sharedObject` was narrowed to a
reward toggle while spawning became inferred. When a change *can't* be a no-op, state the blast
radius explicitly (which runs/datasets it invalidates).

## Config knobs

Three places, in order:

1. `main/cfg/config.yaml` — the knob plus a comment covering semantics, units, the default's
   meaning, and when you'd change it. Multi-line is normal here.
2. `main/cfg/task/<Task>.yaml` — `${resolve_default:<default>,${...knobName}}`, one-line comment
   pointing back to config.yaml.
3. The env — read with `self.cfg["env"].get("knobName", <default>)`, same default as the YAML.

camelCase in YAML, snake_case on the Python attribute. Don't add a knob for a value that has
exactly one correct setting — put it in a table or a constant instead.

## Python environment

- Python 3.8 — no `match`, no `X | Y` unions at runtime, no `dict[str, int]` builtins in
  annotations. `from __future__ import annotations` where hints get in the way.
- numpy 1.23.5.
- **`isaacgym` must be imported before `torch`.** Anything importing `main.dataset` triggers the
  package `__init__`, which registers `mano2dexhand.py` and pulls in isaacgym. To use a helper in
  isolation, load it by path with `importlib.util.spec_from_file_location` and register it in
  `sys.modules` before `exec_module` (dataclasses need the module resolvable by name).
- Modules meant to be imported from both the loaders and the env stay stdlib-only, so they can't
  trip the import order. `object_sets.py` and `my_dataset_utils.py` are both deliberately
  dependency-free.

## Plots

Axis labels and titles in Title Case — `Cap Speed`, `Normalized Time`, `Body-Local X [m]`. Units
stay in their brackets.

## SLURM scripts

Live in `slurm/scitas/` (or `slurm/alps/`), `.run` extension. Header block after the `#SBATCH`
directives explaining what the script does and how to invoke it; keep it short. Arrays use an
`INDICES=(...)` / `CONFIGS=(...)` array indexed by `$SLURM_ARRAY_TASK_ID`, with a
`sleep $((SLURM_ARRAY_TASK_ID * 5))` stagger so concurrent tasks don't contend on startup.

Anything invoking Isaac Gym needs the `LD_LIBRARY_PATH` fix — `gym_38.so` links
`libpython3.8.so.1.0` from the conda env's `lib/`, which `conda activate` does not add. Copy the
block from `retarget_pkl.sh`.

Size `--time` from a measured per-item cost, not a round number: shorter requests backfill sooner.

## Verify, then claim

`bash -n` every shell script and compose-check every Hydra override set before submitting a job —
a config typo costs a GPU allocation. Where a change is checkable offline, check it: asset and mass
changes can be exercised on a CPU `SimParams` with `physx.use_gpu = False`. State what you verified
and how.

Note that SLURM `.out` files on this cluster go stale mid-job; `runs/<experiment>/` is the reliable
progress signal.
