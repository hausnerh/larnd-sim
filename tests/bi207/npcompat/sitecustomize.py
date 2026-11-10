# Compatibility shims for running the current ndlar_flow clone under the sim2x2 container, loaded
# automatically because this directory is on PYTHONPATH (CPython imports `sitecustomize` at
# interpreter startup). Two unrelated fixes live here; both are no-ops outside the bi207 flow step.
#
# (1) numpy >= 1.24 removed the long-deprecated scalar aliases (np.bool, np.int, np.float, ...).
#     The ndlar_flow clone still uses them (e.g. np.bool in a type annotation in
#     proto_nd_flow/reco/charge/raw_event_builder.py), which AttributeErrors under the container's
#     newer numpy. We restore them to the plain builtins they were deprecated in favor of -- which
#     is exactly what those annotations mean -- so h5flow can import the ndlar_flow modules. No-op
#     on any numpy that still defines the aliases.
#
# (2) ndlar_flow's get_anchor_unix_ts() hangs forever on pure-MC input (see the block below).
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
# Real PACMAN data has many (>30) type-4 packets stamped with each unix-second, so it breaks on the
# first candidate. A pure larnd-sim MC file has no real PPS/unix-second structure -- its type-4
# packets carry distinct small tick values -- so NO value ever has >30 of them. itertools.count then
# climbs without bound, re-masking the whole buffer each pass, and h5flow hangs at
# "Run loop on charge/raw_events 0/1" (a ~13k-packet Bi-207 FSD-cube file hung here for >2 h).
# Upstream has no is_mc guard and no config knob, so we swap in an O(N) equivalent that returns the
# most-populated unix-ts label (the anchor the loop is hunting for) and can never spin. The absolute
# anchor is cosmetic for the charge-calibration analysis, which uses relative tick times.
def _patch_ndlar_flow_anchor():
    try:
        from proto_nd_flow.reco.charge import raw_timestamp_utils as _rtu
    except Exception:
        return  # not running inside ndlar_flow -> nothing to patch

    def _safe_get_anchor_unix_ts(packets, packet_msg_unix_ts, delay_extractor=None, threshold=30):
        u = _np.asarray(packet_msg_unix_ts)
        if u.size == 0:
            return 0
        vals, counts = _np.unique(u, return_counts=True)
        return int(vals[int(_np.argmax(counts))])

    _rtu.get_anchor_unix_ts = _safe_get_anchor_unix_ts
    # raw_event_generator does `from .raw_timestamp_utils import get_anchor_unix_ts`; if it was
    # imported before this shim ran, rebind the name it already captured as well.
    try:
        import sys as _sys
        _gen = _sys.modules.get("proto_nd_flow.reco.charge.raw_event_generator")
        if _gen is not None and hasattr(_gen, "get_anchor_unix_ts"):
            _gen.get_anchor_unix_ts = _safe_get_anchor_unix_ts
    except Exception:
        pass


_patch_ndlar_flow_anchor()
