#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_pixgeom.py -- GROUND-TRUTH probe of the fsd_cube pixel geometry, to settle
# the chip/channel<->physical-pixel mapping once (the hitmap keeps putting the centered source at a Z
# edge). Pure print; loads the fsd_cube config (login node, seconds), no LARND needed.
#
# It replicates larnd-sim's EXACT forward assignment on the CPU:
#   pix_x = int((x - TPC_BORDERS[plane][0][0]) // PIXEL_PITCH)           (pixels_from_track.py)
#   pix_y = int((y - TPC_BORDERS[plane][1][0]) // PIXEL_PITCH)
#   tile_id = TILE_MAP[anode][pix_x//NPT0][pix_y//NPT1]
#   chip,channel = PIXEL_CONNECTION_DICT[rotate_tile((pix_x%NPT0, pix_y%NPT1), tile_id)]   (fee.py)
# and prints, for the KNOWN Bi-207 source (anode center, x=23.81 y=133.92) and a scan across x, the
# resulting (pix_x, pix_y, chip, channel) -- so we can see directly whether the source maps to the
# observed hot pixel (chip 166 channel 43) and how pixel index relates to physical x.
#
# Usage:  python tests/bi207/check_bi207_pixgeom.py
# =============================================================================
import numpy as np


def load():
    from larndsim import consts
    from larndsim.config import get_config
    from larndsim.consts import detector
    cfg = get_config("fsd_cube")

    def one(k):
        v = cfg.get(k)
        return v[0] if isinstance(v, list) else v

    consts.load_properties(one("DET_PROPERTIES"), one("PIXEL_LAYOUT"),
                           one("RESPONSE"), one("SIM_PROPERTIES"))
    return detector


def forward(detector, rotate_tile, x, y, plane=0):
    """Replicate pixels_from_track + fee.py exactly. Returns dict with every intermediate."""
    B = np.asarray(detector.TPC_BORDERS)[plane]
    bx, by = float(B[0][0]), float(B[1][0])          # the FIRST border element, as the kernels use it
    pitch = float(detector.PIXEL_PITCH)
    pix_x = int((x - bx) // pitch)
    pix_y = int((y - by) // pitch)
    npt0, npt1 = int(detector.N_PIXELS_PER_TILE[0]), int(detector.N_PIXELS_PER_TILE[1])
    out = dict(x=x, y=y, bx=bx, by=by, pix_x=pix_x, pix_y=pix_y)
    try:
        tile_x, tile_y = pix_x // npt0, pix_y // npt1
        tile_id = detector.TILE_MAP[0][tile_x][tile_y]
        key = tuple(rotate_tile((pix_x % npt0, pix_y % npt1), tile_id))
        cc = detector.PIXEL_CONNECTION_DICT.get(key)
        out.update(tile_id=tile_id, rot_key=key, chip=(cc[0] if cc else None),
                   channel=(cc[1] if cc else None),
                   orient=list(detector.TILE_ORIENTATIONS[tile_id]))
    except Exception as exc:
        out.update(error=repr(exc))
    return out


def main():
    detector = load()
    from larndsim.fee import rotate_tile
    B = np.asarray(detector.TPC_BORDERS)[0]
    print("=== fsd_cube pixel geometry (raw, unsorted) ===")
    print(f"TPC_BORDERS[0] x={list(B[0])}  y={list(B[1])}  z={list(B[2])}")
    print(f"N_PIXELS={tuple(detector.N_PIXELS)}  N_PIXELS_PER_TILE={tuple(detector.N_PIXELS_PER_TILE)}  "
          f"PIXEL_PITCH={float(detector.PIXEL_PITCH):.4f} cm")
    try:
        tm = np.asarray(detector.TILE_MAP)
        print(f"TILE_MAP shape={tm.shape}  content={detector.TILE_MAP}")
        tid0 = detector.TILE_MAP[0][0][0]
        print(f"tile_id at [0][0][0]={tid0}  TILE_ORIENTATIONS[{tid0}]={list(detector.TILE_ORIENTATIONS[tid0])}")
    except Exception as exc:
        print(f"(TILE_MAP/ORIENTATIONS: {exc!r})")

    print("\n=== forward map for the KNOWN source (anode center, cathode) ===")
    src = forward(detector, rotate_tile, 23.81, 133.92)
    print(f"source (x=23.81, y=133.92): pix_x={src['pix_x']} pix_y={src['pix_y']}  "
          f"-> chip={src.get('chip')} channel={src.get('channel')}  "
          f"(tile_id={src.get('tile_id')}, orient={src.get('orient')}, rot_key={src.get('rot_key')})")
    print(f"   observed hottest LARND pixel = chip 166 channel 43.  MATCH = "
          f"{src.get('chip')==166 and src.get('channel')==43}")
    print("   -> if MATCH, the hot pixel IS the centered source; pixel_index->cm must use the border/flip")
    print("      the same way (the hitmap's x=border0+pix*pitch is then the bug).")

    print("\n=== scan physical x across the plane (y fixed at center 133.92) ===")
    print(f"{'x[cm]':>7} {'pix_x':>6} {'chip':>5} {'chan':>5}")
    for x in (0.5, 5.0, 10.0, 23.81, 35.0, 47.0):
        r = forward(detector, rotate_tile, x, 133.92)
        print(f"{x:7.2f} {r['pix_x']:6d} {str(r.get('chip')):>5} {str(r.get('channel')):>5}")
    print("\nRead: if x=23.81 (center) gives chip/channel 166/43 (the hot pixel) then the SOURCE is the")
    print("hot pixel and the hitmap just needs to label that pixel's physical x as ~23.8 (center), i.e.")
    print("pixel index -> cm must invert the same border convention pixels_from_track used.")


if __name__ == "__main__":
    main()
