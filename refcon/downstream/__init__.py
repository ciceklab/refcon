# REFCON downstream callers (run on REFCON's reference-free CN predictions):
#   markers.py  - confident-normal detection (SCEVAN's marker enrichment on expression)
#   classify.py - call_sample: confident-normal cells -> GMM clustering -> per-cluster label (dispersion ratio,
#                 tau1 = 1.05; pure-sample fallback T_sigma = 0.125)
# Import as a package, e.g. `from refcon.downstream.classify import call_sample`
# (see scripts/classify.py).
