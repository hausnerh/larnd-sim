# numpy >= 1.24 removed the long-deprecated scalar aliases (np.bool, np.int, np.float, ...).
# The current ndlar_flow clone still uses them (e.g. np.bool in a type annotation in
# proto_nd_flow/reco/charge/raw_event_builder.py), which AttributeErrors under the container's
# newer numpy. Putting THIS directory on PYTHONPATH makes CPython auto-import sitecustomize at
# interpreter startup and restore the aliases to the plain builtins they were deprecated in favor
# of -- exactly what those annotations mean -- so h5flow can import the ndlar_flow modules.
#
# Scope: only used when $ND_SRC/tests/bi207/npcompat is explicitly on PYTHONPATH (the bi207_fsd
# flow step). It is a no-op on any numpy that still defines the aliases.
import numpy as _np

for _name, _builtin in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("str", str), ("long", int), ("unicode", str)):
    if not hasattr(_np, _name):
        try:
            setattr(_np, _name, _builtin)
        except Exception:
            pass
