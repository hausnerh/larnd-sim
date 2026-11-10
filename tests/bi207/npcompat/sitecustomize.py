# Compatibility shims for running the current ndlar_flow clone under the sim2x2 container, loaded
# automatically because this directory is on PYTHONPATH (CPython imports `sitecustomize` at
# interpreter startup). Two unrelated fixes live here; both are no-ops outside the bi207 flow step.
#
# (1) numpy >= 1.24 removed the long-deprecated scalar aliases (np.bool, np.int, np.float, ...).
#     The ndlar_flow clone still uses them (e.g. np.bool in a type annotation in
#     proto_nd_flow/reco/charge/raw_event_builder.py), which AttributeErrors under the container's
#     newer numpy. We restore them to the plain builtins they were deprecated in favor of. No-op on
#     any numpy that still defines the aliases.
#
# (2) ndlar_flow's get_anchor_unix_ts() hangs forever on pure-MC input (see the block below).
import sys as _sys

import numpy as _np

for _name, _builtin in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("str", str), ("long", int), ("unicode", str)):
    if not hasattr(_np, _name):
        try:
            setattr(_np, _name, _builtin)
        except Exception:
            pass


# --------------------------------------------------------------------------------------------------
# ndlar_flow RawEventGenerator.next() calls get_anchor_unix_ts() (raw_timestamp_utils.py) once, to
# recover an absolute unix-second anchor from the PACMAN timestamp (type-4) packets. Its core is:
#
#     anchor0 = u[0]
#     for anchor in itertools.count(anchor0):              # UNBOUNDED counter
#         anchor_unix_pkts = p[(u == anchor) & (packet_type == 4)]
#         if len(anchor_unix_pkts) > threshold:  break     # threshold defaults to 30
#
# Real PACMAN data (and dense beam/cosmic MC) has >30 type-4 packets stamped with each unix-second,
# so it breaks on the first candidate. A sparse larnd-sim MC file -- e.g. a Bi-207 calibration
# source: 5000 isolated, low-occupancy decays that larnd-sim's non-beam timing spreads ~100 ms apart
# over ~500 s -- has only ~13 type-4 packets per 1 s PACMAN rollover, never >30, so itertools.count
# climbs without bound and h5flow hangs at "Run loop on charge/raw_events 0/1" (a ~13k-packet file
# hung >2 h; on a login node the CPU watchdog SIGKILLs it -> "Killed"). Upstream has no is_mc guard
# and no config knob, so we swap in an O(N) equivalent returning the most-populated unix-ts label
# (the anchor the loop is hunting for). The absolute anchor is cosmetic for the charge analysis,
# which uses relative tick times.
#
# We patch at the module's REAL import (a meta_path post-import hook), not eagerly here: at
# sitecustomize time (deep in interpreter startup) proto_nd_flow is not yet safely importable, so an
# eager `import` fails silently and the patch never lands. The hook fires when h5flow itself imports
# the module, which always succeeds.
_TARGET = "proto_nd_flow.reco.charge.raw_timestamp_utils"


def _safe_get_anchor_unix_ts(packets, packet_msg_unix_ts, delay_extractor=None, threshold=30):
    u = _np.asarray(packet_msg_unix_ts)
    if u.size == 0:
        anchor = 0
    else:
        vals, counts = _np.unique(u, return_counts=True)
        anchor = int(vals[int(_np.argmax(counts))])
    if not getattr(_safe_get_anchor_unix_ts, "_announced", False):
        _sys.stderr.write("[npcompat] get_anchor_unix_ts: MC bypass active, anchor=%d\n" % anchor)
        _safe_get_anchor_unix_ts._announced = True
    return anchor


def _apply_anchor_patch(mod):
    try:
        mod.get_anchor_unix_ts = _safe_get_anchor_unix_ts
        gen = _sys.modules.get("proto_nd_flow.reco.charge.raw_event_generator")
        if gen is not None and hasattr(gen, "get_anchor_unix_ts"):
            gen.get_anchor_unix_ts = _safe_get_anchor_unix_ts
        _sys.stderr.write("[npcompat] patched ndlar_flow get_anchor_unix_ts (MC anti-hang)\n")
    except Exception as _e:
        _sys.stderr.write("[npcompat] WARNING: could not patch get_anchor_unix_ts: %r\n" % (_e,))


if _TARGET in _sys.modules:
    _apply_anchor_patch(_sys.modules[_TARGET])
else:
    import importlib.abc as _ila
    import importlib.util as _ilu

    class _AnchorPatchFinder(_ila.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            _sys.meta_path.remove(self)                    # avoid recursing into ourselves
            try:
                spec = _ilu.find_spec(name)
            finally:
                _sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return None
            _orig_exec = spec.loader.exec_module

            def _exec_module(module, _orig_exec=_orig_exec):
                _orig_exec(module)
                _apply_anchor_patch(module)

            spec.loader.exec_module = _exec_module
            return spec

    _sys.meta_path.insert(0, _AnchorPatchFinder())
