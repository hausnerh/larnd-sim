# =============================================================================
# mjkramer_shoulder_split.py  --  APPLES-TO-APPLES shoulder decomposition for
# mjkramer's FSDCube data/MC notebook:
#     https://gist.github.com/mjkramer/a00a0dde7dccba948bedb8cbbe448585
#
# These are CELLS TO PASTE INTO THAT NOTEBOOK (they reuse its cached_analysis,
# CUTS, Hist, QBINS/QMIN/QMAX, make_hist_dict, get_q_hist, get_1hitPix_hit_hist,
# SIM_CONFIGS, np, plt), so they run on the notebook's OWN data + sim FLOW files
# with the SAME cuts -- a true apples-to-apples reproduction, not a re-implementation.
# Paste AFTER the cell that defines make_hist_dict / builds q_hists (its cell 11).
#
# Goal: reproduce cell 13's "Hit Q for all dt" (per-hit Q, 50 bins 0-50 ke-,
# density), but SPLIT the hits into single-hit-pixel vs multi-hit-pixel, for both
# data and sim, to see which population carries the mid-Q shoulder.
# =============================================================================

# %% -- the missing counterpart to get_1hitPix_hit_hist: MULTI-hit-pixel hits ----
def get_multiHitPix_hit_hist(key, cuts=CUTS):
    """Every hit whose pixel fired MORE THAN ONCE (the mirror of get_1hitPix_hit_hist)."""
    results = cached_analysis(key, **cuts)
    qs: list[float] = []
    for (file_id, event_id), pixel_data in results.items():
        for pixel_id, hits in pixel_data.items():
            if len(hits) <= 1:            # keep ONLY multi-hit pixels
                continue
            qs.extend([hit.q for hit in hits])
    qs = np.array(qs)
    h = Hist.new.Reg(QBINS, QMIN, QMAX, name='q_multi',
                     label=R'Pixel Q (multi-hit pixels) [ke$^-$]',
                     overflow=False).Double()
    h.fill(qs)
    return h


# %% -- build the three populations for every config (data + sim), reusing loaders ----
q_hists  = make_hist_dict(get_q_hist)                # all hits          (== notebook cell 11)
q1_hists = make_hist_dict(get_1hitPix_hit_hist)      # single-hit-pixel  (notebook cell 8)
qm_hists = make_hist_dict(get_multiHitPix_hit_hist)  # multi-hit-pixel   (new)


# %% -- overlay all / single / multi for ONE config, density-normalized ----
def shoulder_split(key, title=None):
    """Overlay per-hit Q for all hits, single-hit-pixel hits, and multi-hit-pixel hits (each as its
    own density) for one config key (e.g. 'data_prc2' or 'sim_prc2_r3_patch_noDrop'). Whichever
    population's shape carries the mid-Q shoulder is the source of the shoulder."""
    for h, lab in ((q_hists[key], 'all hits'),
                   (q1_hists[key], 'single-hit pixels'),
                   (qm_hists[key], 'multi-hit pixels')):
        s = h.sum()
        (h / s if s else h).plot(label=lab)
    plt.legend(); plt.ylabel('Density'); plt.xlabel(R'Hit Q [ke$^-$]')
    plt.title(title or key)


# %% -- reproduce cell 13's prc2 | prc16 side-by-side, split, for the DATA ----
plt.figure(figsize=(12.8, 4.8))
plt.subplot(121); shoulder_split('data_prc2',  'data prc2 - Q for all dt')
plt.subplot(122); shoulder_split('data_prc16', 'data prc16 - Q for all dt')

# %% -- and for a representative SIM config (edit r3/r4, _noFFE, to taste) ----
plt.figure(figsize=(12.8, 4.8))
plt.subplot(121); shoulder_split('sim_prc2_r3_patch_noDrop',  'sim prc2 r3 - Q for all dt')
plt.subplot(122); shoulder_split('sim_prc16_r3_patch_noDrop', 'sim prc16 r3 - Q for all dt')

# %% -- optional: data-vs-sim for JUST the multi-hit population (is the shoulder
#       mismatch in the multi-hit hits?). Reuses the notebook's own compare(). ----
# compare(qm_hists, 'Multi-hit-pixel Q for all dt', ['data_prc2', 'prc2_r3_patch_noDrop'], density=True)
