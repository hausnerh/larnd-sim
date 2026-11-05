#!/usr/bin/env python3
# =============================================================================
# tests/inspect_flow.py -- dump the dataset tree AND attributes of an HDF5 file
# (an ndlar_flow FLOW file, or a larnd-sim OUTPUT file).
#
# Two uses:
#  1) FLOW files: see which hit levels a file carries (charge/raw_hits,
#     charge/calib_prompt_hits, ...) and their fields, to localize data/MC
#     differences at the FEE vs the reconstruction.
#  2) larnd-sim OUTPUT files (e.g. .../prc2_r4_patch_noDrop/*.hdf5, the ones used
#     as edep): the run config is stored in HDF5 ATTRIBUTES -- a 'configs' group
#     and root attrs GIT_COMMIT / GIT_BRANCH / VERSION (simulate_pixels.py:1782).
#     Attributes are printed by default so the exact config + code commit (incl.
#     farfield_enabled / farfield_mode / dipole mode) can be read straight off disk.
#
# Usage:
#   python tests/inspect_flow.py /path/to/one.hdf5                 # charge/ tree + attrs
#   python tests/inspect_flow.py '.../prc2_r4_patch_noDrop/*.hdf5' --group /   # whole file
#   python tests/inspect_flow.py file.hdf5 --rows 3                # + first rows of each dataset
#   python tests/inspect_flow.py file.hdf5 --no-attrs             # schema only
# =============================================================================
import argparse
import glob
import os
import sys

ATTR_MAXLEN = 2000     # truncate very long attribute values (e.g. an embedded yaml blob)


def resolve_one(path):
    """Accept a file, a glob, or a directory; return the first matching file."""
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


def _fmt(v):
    s = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
    s = s.replace("\n", "\\n")
    return s if len(s) <= ATTR_MAXLEN else s[:ATTR_MAXLEN] + f" ... [{len(s)} chars]"


def print_attrs(obj, pre, show):
    if not show:
        return
    try:
        items = list(obj.attrs.items())
    except Exception:  # noqa: BLE001 -- diagnostic, never fatal
        items = []
    for k, v in items:
        print(f"{pre}    @{k} = {_fmt(v)}")


def describe(name, obj, rows, attrs, pre=""):
    fields = obj.dtype.names
    if fields:
        print(f"{pre}{name}  shape={obj.shape}  fields={fields}")
    else:
        print(f"{pre}{name}  shape={obj.shape}  dtype={obj.dtype}")
    print_attrs(obj, pre, attrs)
    if rows and obj.shape and obj.shape[0]:
        try:
            for r in obj[:rows]:
                print(f"{pre}      {r}")
        except Exception as e:  # noqa: BLE001
            print(f"{pre}      (could not read rows: {e})")


def walk(group, rows, attrs, pre=""):
    for k in group:
        obj = group[k]
        path = pre + k
        if hasattr(obj, "dtype"):          # dataset (groups have no .dtype)
            describe(path, obj, rows, attrs, pre="  ")
        else:                              # subgroup -> recurse
            print(f"  {path}/")
            print_attrs(obj, "  ", attrs)
            walk(obj, rows, attrs, pre=path + "/")


def main():
    ap = argparse.ArgumentParser(description="Dump the tree + attributes of an HDF5 file.")
    ap.add_argument("path", help="file, a glob, or a directory (first match is used)")
    ap.add_argument("--group", default="charge",
                    help="top group to dump (default 'charge'; use '/' for the whole file)")
    ap.add_argument("--rows", type=int, default=0,
                    help="also print the first N rows of each dataset (default 0 = schema only)")
    ap.add_argument("--no-attrs", dest="attrs", action="store_false",
                    help="suppress HDF5 attributes (default: show them -- the config lives there)")
    ap.set_defaults(attrs=True)
    args = ap.parse_args()

    path = resolve_one(args.path)
    with open_ro(path) as h:
        print(f"# file: {path}")
        print(f"# top-level groups: {list(h.keys())}")
        print("# root attrs:")
        print_attrs(h, "", args.attrs)
        print()
        grp = h if args.group in ("/", "") else h.get(args.group)
        if grp is None:
            sys.exit(f"!! group '{args.group}' not in this file; top-level keys: {list(h.keys())}")
        print("/" if grp is h else args.group + "/")
        print_attrs(grp, "", args.attrs)
        walk(grp, args.rows, args.attrs, pre="")


if __name__ == "__main__":
    main()
