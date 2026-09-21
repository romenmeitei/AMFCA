# ============================================================
# M01 OPTICAL BACKGROUND COMPARISON v1.0
# Run in Google Colab after the completed M01 concurrence audit.
# Uses the SAME saved rasters, grid, buffers and validity masks.
# No Earth Engine calls, new thresholds, or raster edits.
# ============================================================
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile

# Pin the successful audit reported in this conversation.
BASE_AUDIT = Path(
    '/content/drive/MyDrive/Lhende_2026_Multisensor_Concurrence/'
    'M01_audit_20260911T072517612102Z'
)
SAVE_TO_DRIVE = True

for package in ['numpy', 'pandas', 'rasterio', 'shapely', 'pyproj']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as geometry_transform
from pyproj import CRS, Transformer
from IPython.display import display


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator > 0 else np.nan


def compare_on_common_pixels(codes, rms, zones, joint, pixel_km2, thresholds):
    """Descriptive area/rate comparisons only; no pixel-independence tests."""
    class_labels = {
        0: 'No SAR interval exceeded threshold',
        1: 'Pre-event control exceedance only',
        2: 'Event and pre-event control exceedance',
        3: 'Event interval exceedance only',
    }
    class_rows, comparison_rows = [], []
    references = {
        'all_other_SAR_classes_0_1_2': [0, 1, 2],
        'no_SAR_exceedance_class0': [0],
        'event_and_control_exceedance_class2': [2],
    }
    for zone_code, zid in [(5, 'R05'), (6, 'R06')]:
        z_joint = (zones == zone_code) & joint
        candidate = z_joint & (codes == 3)
        n_candidate = int(np.count_nonzero(candidate))
        for threshold in thresholds:
            optical_high = rms >= threshold
            h_candidate = int(np.count_nonzero(candidate & optical_high))
            p_candidate = finite_ratio(h_candidate, n_candidate)
            for sar_class, label in class_labels.items():
                mask = z_joint & (codes == sar_class)
                n = int(np.count_nonzero(mask))
                h = int(np.count_nonzero(mask & optical_high))
                class_rows.append({
                    'zone_id': zid, 'rms_threshold': threshold,
                    'sar_class': sar_class, 'sar_class_label': label,
                    'jointly_observed_pixel_count': n,
                    'optical_exceedance_pixel_count': h,
                    'jointly_observed_area_km2': n * pixel_km2,
                    'optical_exceedance_area_km2': h * pixel_km2,
                    'optical_exceedance_pct': 100 * finite_ratio(h, n),
                })
            for reference_name, reference_codes in references.items():
                reference = z_joint & np.isin(codes, reference_codes)
                n_reference = int(np.count_nonzero(reference))
                h_reference = int(np.count_nonzero(reference & optical_high))
                p_reference = finite_ratio(h_reference, n_reference)
                if n_candidate == 0 or n_reference == 0:
                    direction = 'not_estimable_missing_group'
                    ratio = np.nan
                elif h_reference == 0:
                    direction = ('higher_class3_rate_reference_zero'
                                 if h_candidate > 0 else 'both_rates_zero')
                    ratio = np.nan  # Do not add a pseudocount or report infinity.
                else:
                    ratio = p_candidate / p_reference
                    direction = ('higher_observed_rate_in_class3' if ratio > 1 + 1e-12
                                 else 'lower_observed_rate_in_class3' if ratio < 1 - 1e-12
                                 else 'equal_observed_rates')
                comparison_rows.append({
                    'zone_id': zid, 'rms_threshold': threshold,
                    'reference_group': reference_name,
                    'class3_joint_pixels': n_candidate,
                    'class3_optical_high_pixels': h_candidate,
                    'class3_optical_low_pixels': n_candidate - h_candidate,
                    'reference_joint_pixels': n_reference,
                    'reference_optical_high_pixels': h_reference,
                    'reference_optical_low_pixels': n_reference - h_reference,
                    'class3_joint_area_km2': n_candidate * pixel_km2,
                    'reference_joint_area_km2': n_reference * pixel_km2,
                    'class3_optical_exceedance_pct': 100 * p_candidate,
                    'reference_optical_exceedance_pct': 100 * p_reference,
                    'difference_percentage_points': 100 * (p_candidate - p_reference),
                    'optical_exceedance_rate_ratio': ratio,
                    'descriptive_direction': direction,
                    'inference_status': 'descriptive_unadjusted_no_p_value',
                })
    return pd.DataFrame(class_rows), pd.DataFrame(comparison_rows)


def run_background_comparison(base_audit=BASE_AUDIT, save_to_drive=SAVE_TO_DRIVE,
                              output_root='/content/Lhende_2026_concurrence'):
    """Recover the exact successful audit, verify it, then compare optical rates."""
    base = Path(base_audit)
    drive_root = Path('/content/drive/MyDrive')
    if (save_to_drive or str(base).startswith('/content/drive/')) and not drive_root.is_dir():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    if not base.is_dir():
        raise FileNotFoundError(f'Audit folder not found: {base}. No alternate version was selected.')

    needed = [
        'M01_method_and_inputs.json', 'M01_observation_coverage.csv',
        'M01_threshold_sensitivity.csv', 'M01_concurrence_RMS008_audited.tif',
        'river_reference_buffers_500m_used.geojson', 'output_checksums.csv',
    ]
    for filename in needed:
        if not (base / filename).is_file():
            raise FileNotFoundError(f'Missing audited output: {base / filename}')

    recorded_outputs = pd.read_csv(base / 'output_checksums.csv')
    expected_output_hashes = dict(zip(recorded_outputs['file'], recorded_outputs['sha256']))
    for filename in needed[:-1]:
        expected_hash = expected_output_hashes.get(filename)
        if not expected_hash or sha256_file(base / filename) != expected_hash:
            raise ValueError(f'Audited-output checksum failed: {filename}. Stop; do not mix versions.')

    meta = json.loads((base / 'M01_method_and_inputs.json').read_text(encoding='utf-8'))
    thresholds = sorted(set(float(t) for t in meta['additional_RMS_sensitivity_thresholds']))
    if not all(np.isfinite(t) and t >= 0 for t in thresholds) or 0.08 not in thresholds:
        raise ValueError('Unexpected recorded threshold configuration.')
    roles = {item['role']: item for item in meta['inputs']}
    if not all(role in roles for role in ['sar', 'optical', 'zones']):
        raise ValueError('The base audit does not record all required source roles.')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(output_root) / ('M01_background_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    cache = out / 'working_inputs'
    cache.mkdir()
    local = {}
    for role in ['sar', 'optical']:
        source = Path(roles[role]['source_path'])
        if not source.is_file():
            raise FileNotFoundError(f'Recorded input is missing: {source}. No replacement was assumed.')
        target = cache / (role + source.suffix.lower())
        shutil.copyfile(source, target)
        if sha256_file(target) != roles[role]['sha256']:
            raise ValueError(f'{role}: input differs from the successful M01 audit.')
        local[role] = target
    zones_path = base / 'river_reference_buffers_500m_used.geojson'
    if sha256_file(zones_path) != roles['zones']['sha256']:
        raise ValueError('Saved buffer geometry differs from the audited input geometry.')

    # Replicate the original categorical input and validity convention.
    with rasterio.open(local['sar']) as ds:
        t = ds.transform
        if (ds.count != 1 or ds.crs != rasterio.crs.CRS.from_epsg(32645) or
            not np.allclose([t.a, t.b, t.d, t.e], [20, 0, 0, -20], atol=1e-8, rtol=0)):
            raise ValueError('Expected original one-band 20 m EPSG:32645 S01 raster.')
        if ds.width * ds.height > 25_000_000:
            raise ValueError('Unexpectedly large SAR extent.')
        if ds.scales[0] != 1 or ds.offsets[0] != 0:
            raise ValueError('Unexpected categorical scale or offset.')
        arr = ds.read(1, masked=True)
        codes = np.asarray(arr.data, dtype=np.float64)
        sar_valid = (~np.ma.getmaskarray(arr) & np.isfinite(codes) & (codes != 255))
        if np.any(sar_valid & ~np.isin(codes, [0, 1, 2, 3])):
            raise ValueError('Unexpected valid SAR category.')
        ref = dict(crs=ds.crs, transform=t, width=ds.width, height=ds.height,
                   bounds=tuple(ds.bounds))
    optical_band = int(meta['raster_metadata']['optical']['used_band'])
    with rasterio.open(local['optical']) as ds:
        rel = (~ds.transform) * t
        if (ds.crs != ref['crs'] or
            not np.allclose([rel.a, rel.b, rel.d, rel.e], [1, 0, 0, 1], atol=1e-7, rtol=0) or
            not np.allclose([rel.c, rel.f], np.round([rel.c, rel.f]), atol=1e-6, rtol=0)):
            raise ValueError('Input pixel grids differ; no resampling was performed.')
        w = Window(int(round(rel.c)), int(round(rel.f)), ref['width'], ref['height'])
        arr = ds.read(optical_band, window=w, boundless=True, masked=True)
        raw = np.asarray(arr.data, dtype=np.float64)
        optical_valid = ~np.ma.getmaskarray(arr) & np.isfinite(raw)
        if ds.nodatavals[optical_band - 1] is not None:
            optical_valid &= raw != ds.nodatavals[optical_band - 1]
        rms = raw * ds.scales[optical_band - 1] + ds.offsets[optical_band - 1]
        optical_valid &= (rms != -9999) & np.isfinite(rms)
        if np.any(optical_valid & (rms < 0)):
            raise ValueError('Negative valid spectral RMS value.')

    # Same pixel-centre inclusion; shared boundary pixels assigned to R05.
    geo = json.loads(zones_path.read_text(encoding='utf-8-sig'))
    if geo.get('type') != 'FeatureCollection' or len(geo.get('features', [])) != 2:
        raise ValueError('Expected two saved river buffers.')
    zone_crs = CRS.from_user_input(
        geo.get('crs', {}).get('properties', {}).get('name') or 'EPSG:4326')
    project = Transformer.from_crs(zone_crs, 32645, always_xy=True)
    projected = {}
    for feature in geo['features']:
        props = feature['properties']
        zid = props['zone_id']
        if (zid not in {'R05', 'R06'} or zid in projected or
            int(props.get('half_width_m', -1)) != 500 or
            props.get('geometry_version') != 'river_reference_candidate_v1'):
            raise ValueError('Unexpected river-buffer identifiers or version.')
        g = shape(feature['geometry'])
        if g.is_empty or not g.is_valid or g.geom_type not in {'Polygon', 'MultiPolygon'}:
            raise ValueError(f'{zid}: invalid polygon; no automatic repairs applied.')
        g = geometry_transform(project.transform, g)
        if not g.is_valid or g.difference(box(*ref['bounds'])).area > 400:
            raise ValueError(f'{zid}: invalid projected geometry or clipped buffer.')
        projected[zid] = g
    if projected['R05'].intersection(projected['R06']).area > 1:
        raise ValueError('Unexpected polygon overlap.')
    zones = np.zeros((ref['height'], ref['width']), dtype=np.uint8)
    for zone_code, zid in [(5, 'R05'), (6, 'R06')]:
        z = rasterize([(mapping(projected[zid]), 1)], out_shape=zones.shape,
                      transform=t, fill=0, all_touched=False, dtype='uint8') > 0
        zones[z & (zones == 0)] = zone_code
    inside = zones > 0
    joint = inside & sar_valid & optical_valid
    pixel_km2 = abs(t.a * t.e - t.b * t.d) / 1e6

    # Must reproduce prior validity, SAR areas, M01 pixels, and threshold counts.
    expected_coverage = pd.read_csv(base / 'M01_observation_coverage.csv').set_index('zone_id')
    expected_sensitivity = pd.read_csv(base / 'M01_threshold_sensitivity.csv')
    reproduction = []
    for zone_code, zid in [(5, 'R05'), (6, 'R06')]:
        z = zones == zone_code
        c3 = z & sar_valid & (codes == 3)
        checks = {
            'buffer_grid_area_km2': np.count_nonzero(z) * pixel_km2,
            'sar_observed_km2': np.count_nonzero(z & sar_valid) * pixel_km2,
            'jointly_observed_km2': np.count_nonzero(z & joint) * pixel_km2,
            'sar_class3_total_km2': np.count_nonzero(c3) * pixel_km2,
            'sar_class3_optically_assessable_km2': np.count_nonzero(c3 & optical_valid) * pixel_km2,
        }
        for key, observed in checks.items():
            if not np.isclose(observed, expected_coverage.loc[zid, key], atol=1e-8, rtol=0):
                raise ValueError(f'{zid}/{key}: previous coverage was not reproduced.')
        for threshold in thresholds:
            previous = expected_sensitivity[
                (expected_sensitivity['zone_id'] == zid) &
                np.isclose(expected_sensitivity['rms_threshold'], threshold)]
            if len(previous) != 1:
                raise ValueError(f'{zid}/{threshold}: base sensitivity row is missing/duplicated.')
            pixels = int(np.count_nonzero(c3 & optical_valid & (rms >= threshold)))
            if pixels != int(previous.iloc[0]['concurrence_pixel_count']):
                raise ValueError(f'{zid}/{threshold}: previous concurrence was not reproduced.')
        reproduction.append({'zone_id': zid, 'prior_coverage_and_threshold_counts': 'PASS'})
    with rasterio.open(base / 'M01_concurrence_RMS008_audited.tif') as ds:
        if (ds.crs != ref['crs'] or ds.transform != t or ds.shape != zones.shape):
            raise ValueError('Audited M01 is not on the same grid.')
        audited = ds.read(1, masked=True)
        previous_valid = ~np.ma.getmaskarray(audited)
        expected_binary = ((codes == 3) & (rms >= 0.08)).astype(np.uint8)
        if np.any(previous_valid != joint) or np.any(audited.data[joint] != expected_binary[joint]):
            raise ValueError('Audited M01 validity or binary pixels differ.')
    print('REPRODUCTION PASS: same inputs, masks, pixels, zones and M01 results.')

    classes, comparisons = compare_on_common_pixels(codes, rms, zones, joint, pixel_km2, thresholds)
    primary = comparisons[
        np.isclose(comparisons['rms_threshold'], 0.08) &
        (comparisons['reference_group'] == 'all_other_SAR_classes_0_1_2')].copy()
    shown = ['zone_id', 'class3_optical_exceedance_pct',
             'reference_optical_exceedance_pct', 'difference_percentage_points',
             'optical_exceedance_rate_ratio', 'descriptive_direction']
    print('\nOPTICAL EXCEEDANCE: SAR CLASS 3 VERSUS OTHER SAR CLASSES — RMS >= 0.08')
    with pd.option_context('display.max_columns', None):
        display(primary[shown].round(4))
    print('\nOPTICAL EXCEEDANCE BY ORIGINAL SAR CLASS — RMS >= 0.08')
    with pd.option_context('display.max_columns', None):
        display(classes[np.isclose(classes['rms_threshold'], 0.08)][[
            'zone_id', 'sar_class', 'jointly_observed_area_km2',
            'optical_exceedance_area_km2', 'optical_exceedance_pct']].round(4))

    tables = {
        'optical_background_comparison_primary.csv': primary,
        'optical_background_comparison_all_thresholds.csv': comparisons,
        'optical_exceedance_by_SAR_class.csv': classes,
        'M01_reproduction_check.csv': pd.DataFrame(reproduction),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    background_meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'base_audit': str(base), 'base_audit_metadata_sha256': sha256_file(base / needed[0]),
        'input_hashes': {r: roles[r]['sha256'] for r in ['sar', 'optical', 'zones']},
        'rms_thresholds': thresholds, 'primary_threshold': 0.08,
        'primary_reference': 'Classes 0, 1, and 2 combined within the SAME jointly observed reach buffer',
        'additional_references': ['Class 0 only', 'Class 2 only'],
        'missing_data': 'Excluded using exactly the successful M01 validity masks',
        'interpretation': 'Unadjusted descriptive spatial association, NOT causal attribution or significance',
        'rate_ratio': 'Optical exceedance rate in class3 divided by rate in the reference group',
        'zero_reference_rate': 'Ratio undefined; no pseudocount or infinity inserted',
        'limitations': [
            'Reference pixels are not verified unaffected sites or matched terrain controls.',
            'River distance, land cover, slope and observation geometry may confound rates.',
            'Neighbouring pixels are spatially dependent; no independent-pixel p-values are calculated.',
            'Optical temporal control and later SAR persistence are not assessed here.',
            'Observed-subset conclusions do not extrapolate to optical gaps, especially R06.',
            'No new thresholds, geometry edits, classifications or spatial filtering were introduced.',
        ],
        'software': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                     'rasterio': rasterio.__version__, 'GDAL': rasterio.__gdal_version__},
    }
    (out / 'background_comparison_metadata.json').write_text(
        json.dumps(background_meta, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame([{'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                  for p in sorted(out.iterdir()) if p.is_file()]).to_csv(
        out / 'output_checksums.csv', index=False)
    with zipfile.ZipFile(out / 'M01_background_comparison_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix != '.zip':
                archive.write(p, arcname=p.name)
    destination = None
    if save_to_drive:
        destination = drive_root / 'Lhende_2026_Multisensor_Concurrence' / out.name
        destination.mkdir(parents=True, exist_ok=False)
        for p in sorted(out.iterdir()):
            if p.is_file():
                target = destination / p.name
                shutil.copyfile(p, target)
                if sha256_file(target) != sha256_file(p):
                    raise IOError(f'Drive output checksum mismatch: {p.name}')
        print('\nSaved and checksum-verified on Drive:', destination)
    else:
        print('\nLocal output:', out)
    print('No old files or rasters were changed. No Earth Engine tasks were submitted.')
    print('These are descriptive rates, not statistical significance or event-damage validation.')
    globals().update(m01_background_primary_df=primary, m01_background_comparisons_df=comparisons,
                     m01_optical_by_sar_class_df=classes, M01_BACKGROUND_OUTPUT_DIR=out,
                     M01_BACKGROUND_DRIVE_DIR=destination)
    return primary, comparisons, classes


if __name__ == '__main__':
    run_background_comparison()
