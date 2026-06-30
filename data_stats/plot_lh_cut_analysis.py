"""Plot left-hand wrist + LH-object trajectories (raw capture vs. retargeted) for a set
of MyDataset capping demos, and auto-detect the terminal "left hand moves quickly in one
direction" motion that should be cut from the end of each demo.

For each demo it writes vis_traj_outputs/lh_cut_analysis/<stem>.png and prints a proposed
[cut_start, T-1] frame range. The cut is detected from the RAW left-wrist speed (the true
captured motion): the last contiguous high-speed run that reaches into the final third of
the clip.

Run:
    python data_stats/plot_lh_cut_analysis.py
"""
import os, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data/my_dataset"
RETARGET_LH = "data/retargeting/my_dataset/mano2inspire_lh"
OUT_DIR = "vis_traj_outputs/lh_cut_analysis"
FPS = 60.0

DEMOS = [
    "m_161528", "m_161551", "m_161610",                                       # cap_1
    "m_170342", "m_170401", "m_170418", "m_170435", "m_170454", "m_170509",   # cap_2
    "m_170527", "m_170541", "m_170556", "m_170612",                           # cap_3
    "m_170639", "m_170654", "m_170708",                                       # cap_4
    "m_170726", "m_170741", "m_170753", "m_170805",                           # cap_5
]


def find_one(directory, suffix_pat):
    hits = [p for p in glob.glob(os.path.join(directory, suffix_pat)) if "_original" not in p]
    assert len(hits) == 1, f"expected 1 match for {suffix_pat} in {directory}, got {hits}"
    return hits[0]


def full_path(p):
    """Prefer the untrimmed *_original backup so plots show the full trajectory + the cut."""
    root, ext = os.path.splitext(p)
    orig = root + "_original" + ext
    return orig if os.path.exists(orig) else p


def interp_nan(arr):
    """Linearly interpolate NaNs along axis 0 (per column). Returns (filled, n_nan_rows)."""
    arr = np.asarray(arr, dtype=np.float64).copy()
    n_nan = int(np.isnan(arr).any(axis=1).sum())
    t = np.arange(len(arr))
    for c in range(arr.shape[1]):
        col = arr[:, c]
        bad = np.isnan(col)
        if bad.all():
            continue
        if bad.any():
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
        arr[:, c] = col
    return arr, n_nan


def speed(pos):
    """Per-frame speed (m/s) from [T,3] positions; length T (first frame duplicated)."""
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS
    return np.concatenate([d[:1], d])


def detect_terminal_cut(wrist_pos, fps=FPS, smooth=5, onset_thr=0.15, min_run=3, peak_min=0.5):
    """Cut = the FIRST frame of the terminal LH 'retract': the earliest frame (in the latter
    part of the clip) where the smoothed leftward velocity `v_s` rises above `onset_thr` AND
    stays above it for at least `min_run` consecutive frames (sustained motion in the same
    'left' direction), within a run that peaks above `peak_min` (a real retract, not a blip).

    Requiring `min_run` consecutive frames of similar leftward velocity prevents a single
    spike from triggering a false cut. Because the cut is the first frame above `onset_thr`,
    it lies exactly on the threshold line in the plot (i.e. the cut == the plotted metric/
    threshold criterion). 'left' is data-driven: the unit direction of the wrist's net
    end-of-trajectory displacement; sideways capping wiggles project to ~0 on it.

    Returns (cut | None, v_s[T], thr, left_dir) — v_s is the smoothed signal that is plotted,
    thr is the onset threshold, and the cut sits on their crossing."""
    wrist_pos = np.asarray(wrist_pos, dtype=np.float64)
    T = len(wrist_pos)
    # 'left' = net terminal displacement (plateau over the first half -> final position)
    ref = np.median(wrist_pos[: max(1, T // 2)], axis=0)
    disp = wrist_pos[-1] - ref
    norm = np.linalg.norm(disp)
    left_dir = disp / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])
    # signed speed along left_dir [T] (positive = moving toward 'left')
    vel = np.diff(wrist_pos, axis=0) * fps                       # [T-1, 3]
    v_left = np.concatenate([[0.0], vel @ left_dir])             # [T]
    v_s = np.convolve(v_left, np.ones(smooth) / smooth, mode="same") if smooth > 1 else v_left
    thr = onset_thr
    # first sustained leftward run in the latter part: v_s stays > thr for >= min_run frames
    # AND the run peaks above peak_min (confirms a real retract, not a small drift / blip).
    t = int(0.4 * T)
    while t < T - min_run:
        if np.all(v_s[t:t + min_run] > thr):
            e = t + min_run                       # extend to the end of the above-thr run
            while e < T and v_s[e] > thr:
                e += 1
            if v_s[t:e].max() >= peak_min:
                return t, v_s, thr, left_dir      # t is the onset (v_s[t-1] <= thr < v_s[t])
            t = e                                  # this run is too weak -> skip past it
        else:
            t += 1
    return None, v_s, thr, left_dir


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for demo in DEMOS:
        raw_path = full_path(find_one(DATA_DIR, f"*{demo}.pkl"))
        ret_path = full_path(find_one(RETARGET_LH, f"*{demo}_lh.pkl"))
        stem = os.path.splitext(os.path.basename(raw_path))[0].replace("_original", "")

        raw = pickle.load(open(raw_path, "rb"))
        ret = pickle.load(open(ret_path, "rb"))

        lw_raw, n_nan_w = interp_nan(raw["hands"]["left"]["wrist_pos"])      # [T,3]
        lobj, n_nan_o = interp_nan(raw["obj_transf"]["bottle_body"][:, :3, 3])  # [T,3] LH object
        lw_ret = np.asarray(ret["opt_wrist_pos"], dtype=np.float64)          # [T,3]
        T = len(lw_raw)
        t = np.arange(T)

        sp_raw = speed(lw_raw)
        sp_ret = speed(lw_ret)
        sp_obj = speed(lobj)

        cut_start, v_s, thr, left_dir = detect_terminal_cut(lw_raw)
        med = float(np.median(sp_raw))

        # ---- figure: 2 rows x 2 cols ----
        fig, ax = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle(f"{stem}   (T={T} frames, {T/FPS:.2f}s)   LH wrist + LH object", fontsize=13)
        axislabels = ["x", "y", "z"]
        colors = ["tab:red", "tab:green", "tab:blue"]

        # (0,0) LH wrist position: raw solid, retargeted dashed
        for c in range(3):
            ax[0, 0].plot(t, lw_raw[:, c], color=colors[c], label=f"raw {axislabels[c]}")
            ax[0, 0].plot(t, lw_ret[:, c], color=colors[c], ls="--", alpha=0.7,
                          label=f"retgt {axislabels[c]}")
        ax[0, 0].set_title("LH wrist position (solid=raw, dashed=retargeted)")
        ax[0, 0].set_xlabel("frame"); ax[0, 0].set_ylabel("m"); ax[0, 0].legend(fontsize=7, ncol=3)

        # (0,1) LH object position
        for c in range(3):
            ax[0, 1].plot(t, lobj[:, c], color=colors[c], label=axislabels[c])
        ax[0, 1].set_title("LH object (bottle_body) position")
        ax[0, 1].set_xlabel("frame"); ax[0, 1].set_ylabel("m"); ax[0, 1].legend(fontsize=8)

        # (1,0) LH wrist velocity along the per-demo 'left'/retract direction; cut = thr crossing
        ax[1, 0].plot(t, v_s, color="purple", label="v_left (smoothed, along retract dir)")
        ax[1, 0].axhline(thr, color="orange", ls=":", label=f"thr={thr:.3f}")
        ax[1, 0].axhline(0.0, color="gray", lw=0.6)
        ax[1, 0].set_title("LH wrist velocity toward 'left'/retract dir  (cut = thr crossing)")
        ax[1, 0].set_xlabel("frame"); ax[1, 0].set_ylabel("m/s"); ax[1, 0].legend(fontsize=8)

        # (1,1) LH object speed
        ax[1, 1].plot(t, sp_obj, color="tab:brown", label="LH obj speed")
        ax[1, 1].set_title("LH object speed"); ax[1, 1].set_xlabel("frame"); ax[1, 1].set_ylabel("m/s")
        ax[1, 1].legend(fontsize=8)

        # mark proposed cut on all panels
        if cut_start is not None:
            for a in ax.flat:
                a.axvspan(cut_start, T - 1, color="red", alpha=0.12)
                a.axvline(cut_start, color="red", ls="--", lw=1.5)
            ax[0, 0].text(cut_start, ax[0, 0].get_ylim()[1], f" cut@{cut_start}",
                          color="red", va="top", fontsize=9)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{stem}.png")
        plt.savefig(out, dpi=110); plt.close(fig)

        kept = cut_start if cut_start is not None else T
        peak_tail = sp_raw[int(0.66 * T):].max()
        summary.append((demo, stem, T, cut_start, kept, n_nan_w, n_nan_o, peak_tail, med))
        print(f"[{demo}] T={T}  cut_start={cut_start}  -> keep [0,{kept-1}] cut [{cut_start},{T-1}]"
              f"  (raw NaN wrist={n_nan_w} obj={n_nan_o}, tail peak={peak_tail:.3f} m/s, med={med:.3f})  {out}")

    print("\n==== SUMMARY (proposed cuts) ====")
    print(f"{'demo':<10}{'T':>5}{'cut_start':>10}{'kept_frames':>13}{'cut_frames':>12}")
    for demo, stem, T, cut_start, kept, *_ in summary:
        cs = "none" if cut_start is None else str(cut_start)
        cf = 0 if cut_start is None else (T - cut_start)
        print(f"{demo:<10}{T:>5}{cs:>10}{kept:>13}{cf:>12}")


if __name__ == "__main__":
    main()
