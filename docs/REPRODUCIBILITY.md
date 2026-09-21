# Reproducibility rules

1. Never overwrite frozen inputs or original detections.
2. Never silently reproject/resample quantitative rasters.
3. Record configuration, UTC timestamps, checksums and software environment.
4. Preserve missing observations as missing.
5. Treat threshold sensitivity as part of the result, not a post-hoc choice.
6. Keep targeted review separate from representative validation.
7. Do not upgrade a candidate to a damage/process label without independent evidence.
8. For exact reproduction, use the frozen scene windows and dated-audit cutoff recorded in `configs/lhende_2026.yaml`.
