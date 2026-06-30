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

DEMOS = ["m_161551", "m_170401", "m_170527", "m_170654", "m_170753"]


def find_one(directory, suffix_pat):
    hits = glob.glob(os.path.join(directory, suffix_pat))
    assert len(hits) == 1, f"expected 1 match for {suffix_pat} in {directory}, got {hits}"
    return hits[0]


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


def detect_terminal_cut(s, gap=25, peak_min=0.4):
    """Given a speed array [T], return the START frame of the terminal fast motion (the
    retract-away burst at the end). The burst oscillates above/below threshold, so we
    threshold, then MERGE fast runs separated by <= `gap` frames into episodes, and take
    the last episode that (a) reaches the final ~5% of the clip and (b) peaks above
    `peak_min` m/s. cut_start = that episode's start, backed off to where the ramp began.
    Returns (cut_start | None, thr, med)."""
    T = len(s)
    med = np.median(s)
    thr = max(5.0 * med, 0.15)
    fast = np.where(s > thr)[0]
    if len(fast) == 0:
        return None, thr, med
    # merge fast-frame indices into episodes separated by gaps > `gap`
    episodes = []
    a = prev = fast[0]
    for idx in fast[1:]:
        if idx - prev <= gap:
            prev = idx
        else:
            episodes.append((a, prev))
            a = prev = idx
    episodes.append((a, prev))
    # terminal episode: ends in the last 5% of the clip and is a real burst
    tail_thresh = T - max(4, int(0.05 * T))
    for (lo, hi) in reversed(episodes):
        if hi >= tail_thresh and s[lo:hi + 1].max() >= peak_min:
            # back off to where the ramp left baseline (speed < 0.5*thr)
            cut = lo
            while cut > 0 and s[cut - 1] > 0.5 * thr:
                cut -= 1
            return cut, thr, med
    return None, thr, med


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for demo in DEMOS:
        raw_path = find_one(DATA_DIR, f"*{demo}.pkl")
        ret_path = find_one(RETARGET_LH, f"*{demo}_lh.pkl")
        stem = os.path.splitext(os.path.basename(raw_path))[0]

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

        cut_start, thr, med = detect_terminal_cut(sp_raw)

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

        # (1,0) LH wrist speed raw vs retargeted
        ax[1, 0].plot(t, sp_raw, color="k", label="raw wrist speed")
        ax[1, 0].plot(t, sp_ret, color="tab:purple", alpha=0.7, label="retgt wrist speed")
        ax[1, 0].axhline(thr, color="orange", ls=":", label=f"thr={thr:.3f}")
        ax[1, 0].set_title("LH wrist speed"); ax[1, 0].set_xlabel("frame"); ax[1, 0].set_ylabel("m/s")
        ax[1, 0].legend(fontsize=8)

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
