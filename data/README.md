# Data

Large satellite rasters are intentionally not redistributed in this repository.

For the exact Lhende 2026 case, the workflow expects the original exported files in Google Drive (paths are in `configs/lhende_2026.yaml`). For a new study, generic mode starts from three co-registered inputs:

1. `sar_control_pattern.tif` — integer codes 0/1/2/3, where 3 is event-only exceedance.
2. `spectral_rms.tif` — optical spectral RMS on the exact same grid.
3. `zones.geojson` — analysis zones containing a `zone_id` field (configurable).

No resampling is performed by the generic concurrence stage.
