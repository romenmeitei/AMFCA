# Standardized generic inputs

## SAR control-pattern raster
Integer values: `0` no interval exceeds threshold; `1` pre-event control only; `2` event and control; `3` event-only. NoData marks unavailable/outside support.

## Optical spectral RMS raster
One-band raster named `spectral_rms`, or the original 14-band optical change export where band 13 is spectral RMS. Must be on the exact SAR grid.

## Zone polygons
GeoJSON with a configurable zone identifier field (default `zone_id`). Pixel-center rasterization is used.

## Generic outputs
- `M01_concurrence.tif`: 0 jointly observed/non-concurrent, 1 concurrence, 255 unknown/outside.
- `M02_joint_optical_support.tif`: 0 SAR observed/optical unavailable, 1 jointly observed, 255 SAR unavailable/outside.
- `candidate_patches.geojson`: 4- or 8-neighbour components grouped separately within each zone.
