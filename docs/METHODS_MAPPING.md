# Manuscript-method to repository mapping

| Manuscript method | Repository stage |
|---|---|
| Multisensor preprocessing | `01_core_gee_lhende_2026.py` |
| SAR–optical concurrence | `02_m01_concurrence_quantification.py` / `amfca.concurrence` |
| Background comparison | `03_m01_background_comparison.py` |
| Candidate patches | `04_candidate_patch_inventory.py` / `amfca.patches` |
| PRE/POST recovery | `05_recover_s2_composites.py` |
| P0001 directional analysis | `06_patch_directional_characterization.py` |
| Dated-scene QA | `07_dated_scene_audit.py` |
| Source-ID reconciliation | `08_scene_inventory_reconciliation.py` |
| Top-20 review | `09_top20_patch_review.py` |
| CEMS/Landsat audit | `10_external_corroboration.py` |
