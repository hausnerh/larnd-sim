#!/usr/bin/env python3
# =============================================================================
# tests/inspect_flow.py -- dump the dataset tree of an ndlar_flow FLOW HDF5.
#
# The data/MC plotter (tests/fsdcube_hitq_datamc.py) reads ONLY
# charge/calib_prompt_hits, i.e. a reconstructed hit level. To check whether the
# data/MC single-hit-Q and hit-multiplicity differences enter at the FEE or in the
# prompt-hit reconstruction, we need the RAW FEE level too (charge/packets, and any
# charge/raw_hits). This prints every dataset under a group with its shape and
# compound field names so we can see which hit levels a file carries and write the
# raw-FEE reader against the real schema.
#
# Usage:
#   python tests/inspect_flow.py /path/to/one.FLOW.hdf5
#   python tests/inspect_flow.py '/pscratch/.../prc2_r4_*/flow/*.FLOW.hdf5'  # first match
#   python tests/inspect_flow.py /path/to/dir               # first *.FLOW.hdf5 under it
#   python tests/inspect_flow.py file.hdf5 --group /        # whole file, not just charge/
#   python tests/inspect_flow.py file.hdf5 --rows 3         # also print the first rows of
#                                                           # each hit-like dataset
# =============================================================================
import argparse
import glob
import os
import sys


def resolve_one(path):
    """Accept a file, a glob, or a directory; return the first matching FLOW file."""
    if any(c in path for c in "*?["):
        hits = sorted(glob.glob(path))
    elif os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", "*.FLOW.hdf5"), recursive=True) or
                      glob.glob(os.path.join(path, "**", "*.hdf5"), recursive=True))
    else:
        hits = [path]
    hits = [p for p in hits if os.path.exists(p)]
    if not hits:
        sys.exit(f"!! no file matched: {path}")
    if len(hits) > 1:
        print(f"# {len(hits)} files matched; inspecting the first:\n#   {hits[0]}\n")
    return hits[0]


def open_ro(path):
    """Open read-only, tolerating older h5py without the locking kwarg (CFS mounts)."""
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def describe(name, obj, rows=0):
    """Print one dataset: path, shape, dtype / compound fields, and optional first rows."""
    fields = obj.dtype.names
    if fields:
        print(f"  {name}  shape={obj.shape}  fields={fields}")
    else:
        print(f"  {name}  shape={obj.shape}  dtype={obj.dtype}")
    if rows and obj.shape and obj.shape[0]:
        try:
            head = obj[:rows]
            for r in head:
                print(f"      {r}")
        except Exception as e:  # noqa: BLE001 -- diagnostic, never fatal
            print(f"      (could not read rows: {e})")


def walk(group, rows=0, pre=""):
    for k in group:
        obj = group[k]
        path = pre + k
        if hasattr(obj, "dtype"):          # dataset (groups have no .dtype)
            describe(path, obj, rows=rows)
        else:                              # subgroup -> recurse
            print(f"  {path}/")
            walk(obj, rows=rows, pre=path + "/")


def main():
    ap = argparse.ArgumentParser(description="Dump the dataset tree of a FLOW HDF5.")
    ap.add_argument("path", help="FLOW file, a glob, or a directory (first match is used)")
    ap.add_argument("--group", default="charge",
                    help="top group to dump (default 'charge'; use '/' for the whole file)")
    ap.add_argument("--rows", type=int, default=0,
                    help="also print the first N rows of each dataset (default 0 = schema only)")
    args = ap.parse_args()

    path = resolve_one(args.path)
    with open_ro(path) as h:
        print(f"# file: {path}")
        print(f"# top-level groups: {list(h.keys())}\n")
        grp = h if args.group in ("/", "") else h.get(args.group)
        if grp is None:
            sys.exit(f"!! group '{args.group}' not in this file; top-level keys: {list(h.keys())}")
        label = "/" if grp is h else args.group + "/"
        print(f"{label}")
        walk(grp, rows=args.rows, pre="")


if __name__ == "__main__":
    main()
