#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_hittruth.py -- GROUND TRUTH: for each LARND data packet, look up its TRUTH
# segment (x,y,z) via mc_packets_assn, and correlate with the packet's (chip, channel). This answers,
# from data (not from my geometry replication), WHERE the hot pixel (chip 166) physically is -- i.e.
# whether the centered source (x=23.81) really maps to chip 166 (=> the decode's cm label is the bug)
# or chip 166 is genuinely at an edge.
#
# Needs the LARND and its input EDEPSIM (segment positions). Login node, seconds.
#   python tests/bi207/check_bi207_hittruth.py --larnd $ND_WORK/bi207_fsd.LARND.h5 \
#       --edep $ND_WORK/bi207_fsd.EDEPSIM.h5
# =============================================================================
import argparse
import numpy as np


def open_ro(path):
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def stat(a, lab):
    a = np.asarray(a, float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return f"{lab}: EMPTY"
    return f"{lab}: min {a.min():7.2f}  med {np.median(a):7.2f}  mean {a.mean():7.2f}  max {a.max():7.2f}  (n={a.size})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--larnd", required=True)
    ap.add_argument("--edep", required=True)
    args = ap.parse_args()

    # --- truth segments from the EDEPSIM (positions), keyed by row index AND by segment_id field ---
    with open_ro(args.edep) as he:
        seg = np.asarray(he["segments"][:]) if "segments" in he else None
    if seg is None:
        raise SystemExit("no segments in the EDEPSIM")
    names = seg.dtype.names

    def axis(ax):
        if ax in names:
            return np.asarray(seg[ax], float)
        if f"{ax}_start" in names and f"{ax}_end" in names:
            return 0.5 * (np.asarray(seg[f"{ax}_start"], float) + np.asarray(seg[f"{ax}_end"], float))
        return None
    sx, sy, sz = axis("x"), axis("y"), axis("z")
    seg_id_field = None
    for cand in ("segment_id", "segment_ids", "id"):
        if cand in names:
            seg_id_field = np.asarray(seg[cand]).astype(np.int64)
            break
    print(f"EDEPSIM segments: n={seg.size}  fields={names}")
    print("  " + stat(sx, "seg x"))
    print("  " + stat(sy, "seg y"))
    if seg_id_field is not None:
        print(f"  segment_id field '{cand}': min {seg_id_field.min()} max {seg_id_field.max()}")

    # --- LARND data packets + their association ---
    with open_ro(args.larnd) as h:
        print(f"\nLARND datasets: {list(h.keys())}")
        p = np.asarray(h["packets"][:])
        from larndsim import consts
        from larndsim.config import get_config
        from larndsim.consts import detector
        cfg = get_config("fsd_cube")
        one = lambda k: (cfg.get(k)[0] if isinstance(cfg.get(k), list) else cfg.get(k))
        consts.load_properties(one("DET_PROPERTIES"), one("PIXEL_LAYOUT"), one("RESPONSE"), one("SIM_PROPERTIES"))
        dtype_pt = int(detector.PACKET_TYPE)
        dmask = p["packet_type"] == dtype_pt
        chip = p["chip_id"][dmask].astype(np.int64)
        chan = p["channel_id"][dmask].astype(np.int64)
        assn = h["mc_packets_assn"]
        print(f"mc_packets_assn fields={assn.dtype.names}")
        a = np.asarray(assn[:])[dmask]
        segids = np.asarray(a["segment_ids"])          # (Ndata, K)
        segids = segids.reshape(segids.shape[0], -1)
        first = segids[:, 0].astype(np.int64)          # leading (highest-fraction) segment per packet

    # --- resolve first-segment -> truth (x,y) by BOTH interpretations, pick the sane one ---
    def lookup_by_index(idx):
        ok = (idx >= 0) & (idx < seg.size)
        tx = np.full(idx.shape, np.nan); ty = np.full(idx.shape, np.nan)
        tx[ok] = sx[idx[ok]]; ty[ok] = sy[idx[ok]]
        return tx, ty, ok
    tx_i, ty_i, ok_i = lookup_by_index(first)
    print(f"\n[interpret segment_ids as ROW INDEX] valid {int(ok_i.sum())}/{first.size}")
    print("  " + stat(tx_i, "truth x of hits"))
    print("  " + stat(ty_i, "truth y of hits"))

    tx, ty = tx_i, ty_i
    if seg_id_field is not None:
        order = np.argsort(seg_id_field)
        sid_sorted = seg_id_field[order]
        pos = np.searchsorted(sid_sorted, first)
        pos = np.clip(pos, 0, sid_sorted.size - 1)
        match = sid_sorted[pos] == first
        idx2 = np.where(match, order[pos], -1)
        tx_f, ty_f, ok_f = lookup_by_index(idx2)
        print(f"[interpret segment_ids as segment_id FIELD] matched {int(ok_f.sum())}/{first.size}")
        print("  " + stat(tx_f, "truth x of hits"))
        print("  " + stat(ty_f, "truth y of hits"))
        if np.isfinite(tx_f).sum() > np.isfinite(tx_i).sum():
            tx, ty = tx_f, ty_f

    # --- the decisive numbers: truth x,y of the HOT pixel (chip 166) vs center (23.81) ---
    print("\n=== where is chip 166 physically (truth x,y of its hits) ? ===")
    for cid in (166, 165, 86):
        m = (chip == cid) & np.isfinite(tx)
        if m.any():
            print(f"  chip {cid}: {int(m.sum())} hits   truth x med {np.median(tx[m]):6.2f}  "
                  f"y med {np.median(ty[m]):7.2f}")
        else:
            print(f"  chip {cid}: no hits with truth")
    print("\nINTERPRETATION: the fsd_cube anode center is x=23.81, y=133.92. If chip 166's hits have")
    print("truth x ~ 23.8, chip 166 IS the center and the hitmap's chip->cm label is the only bug.")
    print("If truth x ~ 0.9 or ~46.5, the charge really is at an edge (a deeper sim/placement issue).")


if __name__ == "__main__":
    main()
