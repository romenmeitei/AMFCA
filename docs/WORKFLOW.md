# Workflow

## Scientific logic

```text
Pre-event controls + event SAR
          |
          v
SAR control pattern (0/1/2/3; class 3 = event-only)
          +
PRE/POST optical reflectance -> spectral RMS
          |
          v
M01 concurrence = SAR class 3 AND spectral RMS >= primary threshold
          |
          v
4-neighbour candidate patches
          |
          +--> local spectral characterization
          +--> dated-scene QA and source reconciliation
          +--> targeted top-N review
          +--> independent-data availability/corroboration audit
```

The exact 2026 study uses the frozen scripts under `workflows/frozen_lhende_2026/`. Generic mode implements the central concurrence + patch steps for arbitrary zones.
