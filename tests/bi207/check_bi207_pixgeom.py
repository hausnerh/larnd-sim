#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_pixgeom.py -- find the CORRECT packet(chip,channel) -> (x,y) chain using ONLY
# larnd-sim's own geometry functions (no hand-rolled index math), and validate it against the known
# truth (check_bi207_hittruth: chip 166's hits sit at the anode center x~23.8). Login node, seconds.
#
# Forward (larndsim/fee.py:248): a pixel (i_x,i_y) fires
#     chip,channel = PIXEL_CONNECTION_DICT[ rotate_tile((i_x % NPT0, i_y % NPT1), tile_id) ]
# so PIXEL_CONNECTION_DICT is keyed by rotate_tile(pixel). The inverse (packet -> pixel) is therefore
#     pixel = rotate_tile( {(chip,channel) -> key}[ (chip,channel) ] , tile_id )     (rotate_tile is an involution)
# and the coordinate is larndsim/detsim.get_pixel_coordinates(pixel2id(i_x,i_y,plane)). We print that
# whole chain for the hot pixels and the forward chip for the center pixel, to see where it breaks.
#
# Usage: python tests/bi207/check_bi207_pixgeom.py
# =============================================================================
import numpy as np

from larndsim import consts
from larndsim.config import get_config
from larndsim.consts import detector
from larndsim.fee import rotate_tile
from larndsim.detsim import get_pixel_coordinates
from larndsim.pixels_from_track import pixel2id, id2pixel


def load():
    cfg = get_config("fsd_cube")
    one = lambda k: (cfg.get(k)[0] if isinstance(cfg.get(k), list) else cfg.get(k))
    consts.load_properties(one("DET_PROPERTIES"), one("PIXEL_LAYOUT"), one("RESPONSE"), one("SIM_PROPERTIES"))


def main():
    load()
    pcd = detector.PIXEL_CONNECTION_DICT                 # {rotate_tile(pixel): (chip, channel)}
    cc2key = {v: k for k, v in pcd.items()}              # (chip,channel) -> that key
    npt = tuple(int(v) for v in detector.N_PIXELS_PER_TILE)
    print(f"N_PIXELS={tuple(int(v) for v in detector.N_PIXELS)}  N_PIXELS_PER_TILE={npt}")
    print(f"TILE_ORIENTATIONS[1]={list(detector.TILE_ORIENTATIONS[1])}  TILE_MAP={detector.TILE_MAP}")
    print(f"PIXEL_CONNECTION_DICT: {len(pcd)} entries; sample {list(pcd.items())[:2]}")

    print("\n=== INVERSE: packet (chip,channel) -> pixel -> get_pixel_coordinates ===")
    print("(truth from check_bi207_hittruth: chip 166 & 165 hits are at x~23.8, y~133.9 = anode center)")
    for cid, ch in [(166, 43), (165, 21), (86, 36)]:
        key = cc2key.get((cid, ch))
        if key is None:
            print(f"  chip {cid} ch {ch}: not in PIXEL_CONNECTION_DICT")
            continue
        phys = rotate_tile(tuple(key), 1)                # un-rotate (involution) -> physical pixel
        x, y = get_pixel_coordinates(pixel2id(int(phys[0]), int(phys[1]), 0))
        print(f"  chip {cid:3d} ch {ch:2d}: PCD key(=ccp value)={tuple(int(v) for v in key)}  "
              f"rotate_tile->pixel={tuple(int(v) for v in phys)}  coord=({x:.2f}, {y:.2f})")

    print("\n=== FORWARD: which chip/channel does each pixel fire, + its coordinate ===")
    for ix, iy in [(64, 40), (2, 39), (125, 39)]:
        key = tuple(rotate_tile((ix, iy), 1))
        x, y = get_pixel_coordinates(pixel2id(ix, iy, 0))
        print(f"  pixel ({ix:3d},{iy:2d}) coord=({x:.2f}, {y:.2f}): rotate_tile->key={key}  "
              f"fee.py chip/channel={pcd.get(key)}")

    print("\nREAD: the INVERSE 'coord' for chip 166/165 should be ~ (23.8, 133.9) if this chain is right.")
    print("If it is ~ (0.9,..) or (46.5,..), compare with FORWARD: whichever pixel's forward chip is 166")
    print("is the real one, and its coord (via get_pixel_coordinates) is where the hitmap must place it.")


if __name__ == "__main__":
    main()
