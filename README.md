# AMFCA — Auditable Multisensor Flood Change Assessment

AMFCA is a reproducibility framework assembled from the validated Google Colab/Python workflows used for the **26 August 2026 Lhende Khola–Bhote Koshi–Trishuli flood study**. It has two goals:

1. **Exact case-study preservation.** Frozen scripts preserve the analytical logic that generated the published/audited Lhende outputs.
2. **Future-study reuse.** A configuration-driven generic mode reproduces the central event-only SAR + optical spectral-RMS concurrence and connected-patch workflow for new studies without silently resampling inputs.

> Candidate/concurrence outputs are remote-sensing surface-change evidence, not automatic flood-damage, erosion, debris, or causal labels.

## Repository layout

```text
configs/                         Case-study and new-event YAML files
notebooks/                       Master Google Colab runner
src/amfca/                       Reusable package code
scripts/                         CLI runner, verifier, core renderer
workflows/frozen_lhende_2026/    Frozen validated study scripts
validation/                      Frozen-script manifest and prior synthetic test reports
tests/                           Offline regression/unit tests
docs/                            Workflow and reproducibility documentation
```

## Quick start — Google Colab

Open `notebooks/AMFCA_Master_Reproducibility_Colab.ipynb`, set your GitHub repository URL after uploading this package, mount Google Drive, and first run:

```bash
python scripts/run_pipeline.py --config configs/lhende_2026.yaml --plan
python scripts/verify_repository.py
```

The exact Lhende downstream pipeline can then be run stage-by-stage. Heavy Earth Engine stages require authentication and should not be launched blindly.

## Exact Lhende 2026 reproduction

The frozen study is split into auditable stages:

1. Core Earth Engine multisensor processing (frozen reference script).
2. R05/R06 M01 concurrence quantification.
3. Descriptive SAR-class optical background comparison.
4. Four-neighbour candidate-patch inventory.
5. Recovery/verification of retained Sentinel-2 PRE/POST composites.
6. Patch directional characterization.
7. Dated Sentinel-2 quality/persistence audit.
8. Scene-inventory/source reconciliation.
9. Top-20 patch review.
10. Priority-patch CEMS/Landsat corroboration and descriptive figures.

Run selected downstream stages, for example:

```bash
python scripts/run_pipeline.py --config configs/lhende_2026.yaml \
  --stages m01_concurrence background_comparison patch_inventory s2_recovery
```

To continue from already completed outputs, put their paths into the `existing_*` fields in `configs/lhende_2026.yaml` and request only later stages.

### Core Earth Engine script

The original core script is frozen and is never edited in place. To create a configured copy:

```bash
python scripts/render_core_from_config.py \
  --config configs/lhende_2026.yaml \
  --output /content/lhende_core_configured.py
python /content/lhende_core_configured.py
```

For exact case-study reproduction, retain the original dates, thresholds, geometry and source inventories. For a new event, review every scientific setting before execution.

## Reuse for a future flood

Copy `configs/template_new_event.yaml`. Generic mode expects **three standardized, exactly co-registered inputs**: the SAR control-pattern raster, optical spectral-RMS raster, and zone polygons. Then run:

```bash
python scripts/run_pipeline.py --config configs/my_new_event.yaml
```

This produces a generic M01 concurrence raster, observation-support raster, threshold table, connected-patch raster, patch inventory and GeoJSON. It is intentionally conservative: if raster grids differ, the pipeline stops rather than silently resampling.

The full upstream sensor preprocessing remains study-specific. `scripts/render_core_from_config.py` can generate a configured copy of the frozen Lhende Earth Engine workflow as a starting point, but users must validate geometry, acquisition windows, SAR orbit logic, thresholds and ancillary data for a new basin.

## Reproducibility safeguards

- Frozen study scripts are checksum-manifested.
- New runs write to separate output directories.
- Generic stages refuse silent raster reprojection/resampling.
- Missing optical coverage remains unknown, not unchanged.
- Priority/top-N review is explicitly targeted, not representative accuracy sampling.
- Source reconciliation is numerical compatibility, not field validation.
- External-data absence is not interpreted as evidence of no impact.

## Verify the repository

```bash
python scripts/verify_repository.py
```

The verifier checks required files, both YAML configurations, frozen-script syntax and checksums, and the offline pytest suite.

## Software environment

Python 3.10–3.12 is recommended. Install with:

```bash
pip install -e .
```

Earth Engine stages additionally require an authorized Google Earth Engine project.

## Citation

A `CITATION.cff.example` file is included. Replace the placeholder author metadata before public release and rename it to `CITATION.cff`.

## License

The repository includes the MIT license for code. Satellite imagery and third-party datasets retain their original licenses and are not redistributed here.

## Archived original Colab notebooks

The final generated notebooks used during the study are preserved under `notebooks/archive/`. The recommended entry point for future runs is `notebooks/AMFCA_Master_Reproducibility_Colab.ipynb`.
