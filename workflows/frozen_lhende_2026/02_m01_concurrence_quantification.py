# ============================================================
# LHENDE: R05/R06 OPTICAL-SAR CONCURRENCE QUANTIFICATION v1.0
# Run this COMPLETE cell in Google Colab.
# Reads the saved GeoTIFFs; no Earth Engine rerun is needed.
# Missing optical observations remain UNKNOWN, not no-change.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

# Optional: put /content/drive/... paths here to avoid uploading.
# Do NOT use Windows I:\\... paths: Colab cannot open that drive.
INPUT_FILES = {
    'sar': '',       # ORIGINAL S01_sar_control_pattern.tif (0,1,2,3)
    'optical': '',   # R03_spectral_rms.tif, or original 14-band R03
    'zones': '',     # river_reference_buffers_500m.geojson
    'qgis_m01': '',  # Optional: M01...Concurrence.tif for comparison
}
SAVE_TO_DRIVE = True
PRIMARY_THRESHOLD = 0.08
RMS_THRESHOLDS = [0.06, 0.08, 0.10]  # Additional sensitivity scenarios
# The exact export run already used in this project:
SAR_RUN_FOLDER = (
    'Lhende_2026_Channel_Audit/'
    'sar_spatial_20260909T104846220104Z'
)

# Install only missing libraries; record versions in the output.
for package in ['rasterio', 'shapely', 'pyproj', 'numpy', 'pandas']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', package
        ])

import numpy as np
import pandas as pd
import rasterio
import shapely
import pyproj
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as transform_geometry
from pyproj import Transformer, CRS as Projection
from IPython.display import display


def run_m01_quantification(input_files=None, save_to_drive=True,
                          output_base='/content/Lhende_2026_concurrence'):
    """File-based calculation; does not modify the input files or old globals."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(output_base) / ('M01_audit_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    warnings = []
    drive_root = Path('/content/drive/MyDrive')

    if save_to_drive and not drive_root.is_dir():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    if save_to_drive and not drive_root.is_dir():
        raise RuntimeError('Drive did not mount. No input data were changed.')

    # --------------------------------------------------------
    # A. Resolve inputs. No broad Drive search or version mixing.
    # --------------------------------------------------------
    paths = {}
    for role, value in (input_files or {}).items():
        if value:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f'{role}: file does not exist: {path}')
            paths[role] = path

    known_run = drive_root / SAR_RUN_FOLDER
    known_inputs = drive_root / 'Lhende_2026_Multisensor_Inputs'
    defaults = {
        'sar': [known_run / 'S01_sar_control_pattern.tif'],
        'zones': [known_run / 'river_reference_buffers_500m.geojson'],
        'optical': [known_inputs / 'R03_spectral_rms.tif',
                    drive_root / 'Lhende_2026_GEE_Exports'
                    / 'R03_optical_change_metrics_RETRY.tif'],
        'qgis_m01': [known_inputs / 'M01_SAR_Class3_OpticalRMS_Concurrence.tif'],
    }
    for role, candidates in defaults.items():
        if role not in paths:
            for candidate in candidates:
                if candidate.is_file():
                    paths[role] = candidate
                    break

    def role_for_file(name):
        stem = re.sub(r'[^a-z0-9]', '', Path(name).stem.lower())
        suffix = Path(name).suffix.lower()
        if suffix in {'.tif', '.tiff'}:
            if stem.startswith('s01sarcontrolpattern'):
                return 'sar'
            if (stem.startswith('r03spectralrms') or
                    stem.startswith('r03opticalchangemetricsretry')):
                return 'optical'
            if stem.startswith('m01') and 'concurrence' in stem:
                return 'qgis_m01'
        if suffix in {'.geojson', '.json'} and stem.startswith('riverreferencebuffers500m'):
            return 'zones'
        return None

    required_roles = ['sar', 'optical', 'zones']
    missing = [role for role in required_roles if role not in paths]
    if missing:
        print('Required inputs not found automatically:', ', '.join(missing))
        print('Upload the missing files. You may also include your QGIS M01 TIFF.')
        print('Use ORIGINAL S01_sar_control_pattern.tif, NOT the Class3 binary copy.')
        from google.colab import files
        uploaded = files.upload()
        upload_dir = out / 'uploaded_inputs'
        upload_dir.mkdir()
        grouped = {}
        for name, content in uploaded.items():
            role = role_for_file(name)
            if role:
                destination = upload_dir / Path(name).name
                destination.write_bytes(content)
                grouped.setdefault(role, []).append(destination)
        del uploaded
        for role, candidates in grouped.items():
            if len(candidates) > 1:
                raise ValueError(f'Multiple {role} inputs uploaded. Supply one per role.')
            if role not in paths:
                paths[role] = candidates[0]

    missing = [role for role in required_roles if role not in paths]
    if missing:
        raise FileNotFoundError(f'Still missing {missing}. No quantitative result created.')
    if len({str(paths[r].resolve()) for r in required_roles}) != 3:
        raise ValueError('The three required inputs must be different files.')
    print('\nINPUTS USED')
    for role, path in paths.items():
        print(f'{role}: {path}')

    # Cache locally to avoid repeated random I/O through mounted Drive.
    cache = out / 'working_inputs'
    cache.mkdir()
    local = {}
    for role, path in paths.items():
        target = cache / (role + path.suffix.lower())
        shutil.copyfile(path, target)
        local[role] = target

    input_metadata = {}
    def record_raster(role, ds, band):
        input_metadata[role] = {
            'crs': str(ds.crs), 'transform': list(ds.transform)[:6],
            'width': ds.width, 'height': ds.height, 'band_count': ds.count,
            'used_band': band, 'description': ds.descriptions[band - 1],
            'dtype': ds.dtypes[band - 1],
            'nodata': (ds.nodatavals[band - 1] if ds.nodatavals[band - 1] is None
                       or np.isfinite(ds.nodatavals[band - 1]) else str(ds.nodatavals[band - 1])),
            'scale': ds.scales[band - 1], 'offset': ds.offsets[band - 1],
        }

    # --------------------------------------------------------
    # B. Native SAR grid and categorical values.
    # --------------------------------------------------------
    with rasterio.open(local['sar']) as ds:
        if ds.count != 1 or ds.crs != rasterio.crs.CRS.from_epsg(32645):
            raise ValueError('S01 must be one-band EPSG:32645.')
        t = ds.transform
        if not np.allclose([t.a, t.b, t.d, t.e], [20, 0, 0, -20], atol=1e-8, rtol=0):
            raise ValueError('S01 is not on the expected north-up 20 m grid.')
        if not (ds.scales[0] == 1 and ds.offsets[0] == 0):
            raise ValueError('Unexpected scale/offset on categorical S01.')
        if ds.width * ds.height > 25_000_000:
            raise ValueError('Unexpectedly large S01. Check the selected file.')
        sar_ma = ds.read(1, masked=True)
        sar_values = np.asarray(sar_ma.data, dtype=np.float64)
        declared_valid = ~np.ma.getmaskarray(sar_ma) & np.isfinite(sar_values)
        # Recognize the explicitly exported category NoData even with a sidecar mask.
        declared_valid &= sar_values != 255
        invalid_codes = declared_valid & ~np.isin(sar_values, [0, 1, 2, 3])
        if invalid_codes.any():
            raise ValueError('S01 contains values other than 0,1,2,3,255/NoData.')
        sar_valid = declared_valid
        if not np.any(sar_valid & (sar_values == 3)):
            raise ValueError('No Class 3 found. Use ORIGINAL S01, not the 0/1 Class3 copy.')
        ref = {'crs': ds.crs, 'transform': t, 'width': ds.width,
               'height': ds.height, 'bounds': tuple(ds.bounds)}
        record_raster('sar', ds, 1)
    shape_hw = (ref['height'], ref['width'])
    pixel_km2 = abs(t.a * t.e - t.b * t.d) / 1e6

    # --------------------------------------------------------
    # C. Read a matching window, without any resampling.
    # A different CRS/pixel lattice stops the primary analysis.
    # --------------------------------------------------------
    def matching_window(path, band, role):
        with rasterio.open(path) as ds:
            record_raster(role, ds, band)
            rel = (~ds.transform) * ref['transform']
            if (ds.crs != ref['crs'] or
                not np.allclose([rel.a, rel.b, rel.d, rel.e], [1, 0, 0, 1], atol=1e-7, rtol=0) or
                not np.allclose([rel.c, rel.f], np.round([rel.c, rel.f]), atol=1e-6, rtol=0)):
                raise ValueError(f'{role}: pixel grids differ. No implicit reprojection was performed.')
            window = Window(int(round(rel.c)), int(round(rel.f)),
                            ref['width'], ref['height'])
            arr = ds.read(band, window=window, boundless=True, masked=True)
            raw = np.asarray(arr.data, dtype=np.float64)
            valid = ~np.ma.getmaskarray(arr) & np.isfinite(raw)
            if ds.nodatavals[band - 1] is not None:
                valid &= raw != ds.nodatavals[band - 1]
            values = raw * ds.scales[band - 1] + ds.offsets[band - 1]
            return values, valid

    with rasterio.open(local['optical']) as ds:
        matches = [i + 1 for i, name in enumerate(ds.descriptions)
                   if name and name.strip().lower() == 'spectral_rms']
        if len(matches) == 1:
            optical_band = matches[0]
        elif ds.count == 1 and 'spectral' in paths['optical'].name.lower():
            optical_band = 1
        else:
            raise ValueError('Cannot verify spectral_rms band. Supply the extracted R03_spectral_rms.tif.')
    rms, optical_valid = matching_window(local['optical'], optical_band, 'optical')
    optical_valid &= rms != -9999
    if np.any(optical_valid & (rms < 0)):
        raise ValueError('Negative valid spectral RMS values: check optical band/NoData.')
    optical_valid &= np.isfinite(rms)
    print('\nGRID CHECK PASS: 20 m EPSG:32645; optical values copied without resampling.')

    # --------------------------------------------------------
    # D. Exact saved R05/R06 reference buffers.
    # Pixel-center inclusion; shared boundary pixels go to R05.
    # --------------------------------------------------------
    geo = json.loads(local['zones'].read_text(encoding='utf-8-sig'))
    features = geo.get('features', [])
    if geo.get('type') != 'FeatureCollection' or len(features) != 2:
        raise ValueError('Expected the two-feature river_reference_buffers_500m.geojson.')
    zone_crs_name = (geo.get('crs', {}).get('properties', {}).get('name') or 'EPSG:4326')
    zone_crs = Projection.from_user_input(zone_crs_name)
    project = Transformer.from_crs(zone_crs, 32645, always_xy=True)
    projected = {}
    for feature in features:
        props = feature.get('properties', {})
        zid = str(props.get('zone_id', ''))
        if (zid not in {'R05', 'R06'} or zid in projected or
            int(props.get('half_width_m', -1)) != 500 or
            props.get('geometry_version') != 'river_reference_candidate_v1'):
            raise ValueError('These are not the audited R05/R06 500 m reference buffers.')
        geometry = shape(feature['geometry'])
        if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {'Polygon', 'MultiPolygon'}:
            raise ValueError(f'{zid}: invalid polygon geometry; no automatic repair applied.')
        if zone_crs.is_geographic and (max(abs(v) for v in geometry.bounds) > 180):
            raise ValueError('GeoJSON CRS/coordinate range is inconsistent.')
        geometry = transform_geometry(project.transform, geometry)
        if not geometry.is_valid:
            raise ValueError(f'{zid}: invalid projected geometry.')
        if geometry.difference(box(*ref['bounds'])).area > 400:
            raise ValueError(f'{zid}: SAR extent clips the buffer. Use the ORIGINAL S01 export.')
        projected[zid] = geometry
    if projected['R05'].intersection(projected['R06']).area > 1.0:
        raise ValueError('R05/R06 polygons overlap by more than 1 square metre.')
    zones = np.zeros(shape_hw, dtype=np.uint8)
    for code, zid in [(5, 'R05'), (6, 'R06')]:
        inside = rasterize([(mapping(projected[zid]), 1)], out_shape=shape_hw,
                           transform=t, fill=0, all_touched=False, dtype='uint8') > 0
        zones[inside & (zones == 0)] = code
    inside = zones > 0
    common = inside & sar_valid & optical_valid
    sar3 = inside & sar_valid & (sar_values == 3)
    if not common.any():
        raise ValueError('There are no jointly observed optical/SAR pixels in these buffers.')

    # --------------------------------------------------------
    # E. Coverage-aware area statistics and threshold sensitivity.
    # Do not count missing optical pixels as optical disagreement.
    # --------------------------------------------------------
    def area(mask):
        return float(np.count_nonzero(mask)) * pixel_km2
    def pct(n, d):
        return 100.0 * n / d if d > 0 else np.nan

    coverage_rows, result_rows = [], []
    thresholds = sorted(set([float(PRIMARY_THRESHOLD), *map(float, RMS_THRESHOLDS)]))
    if any(not np.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError('RMS thresholds must be finite and non-negative.')
    for code, zid in [(5, 'R05'), (6, 'R06')]:
        z = zones == code
        z_sar = z & sar_valid
        z_joint = z & common
        z_class3 = z & sar3
        z_assessable3 = z_class3 & optical_valid
        a_zone, a_sar, a_joint = area(z), area(z_sar), area(z_joint)
        a_class3, a_assessable3 = area(z_class3), area(z_assessable3)
        if a_zone == 0 or a_sar == 0:
            raise ValueError(f'{zid}: no rasterized zone/SAR support.')
        coverage_rows.append({
            'zone_id': zid, 'buffer_geometry_area_km2_utm': projected[zid].area / 1e6,
            'buffer_grid_area_km2': a_zone, 'sar_observed_km2': a_sar,
            'jointly_observed_km2': a_joint,
            'jointly_observed_pct_of_buffer': pct(a_joint, a_zone),
            'sar_class3_total_km2': a_class3,
            'sar_class3_optically_assessable_km2': a_assessable3,
            'sar_class3_optical_unavailable_km2': a_class3 - a_assessable3,
            'optical_assessability_pct_of_class3': pct(a_assessable3, a_class3),
        })
        for threshold in thresholds:
            concurrence = z_assessable3 & (rms >= threshold)
            a_conc = area(concurrence)
            below = area(z_assessable3 & (rms < threshold))
            if not np.isclose(a_conc + below, a_assessable3, atol=1e-8):
                raise RuntimeError('Concurrence/below-threshold partition failed.')
            result_rows.append({
                'zone_id': zid, 'rms_threshold': threshold,
                'concurrence_pixel_count': int(np.count_nonzero(concurrence)),
                'concurrence_km2': a_conc,
                'assessable_class3_below_threshold_km2': below,
                'concurrence_pct_of_assessable_class3': pct(a_conc, a_assessable3),
                'concurrence_pct_of_all_class3_observed_fraction': pct(a_conc, a_class3),
                'concurrence_pct_of_joint_observed_area': pct(a_conc, a_joint),
                'concurrence_pct_of_buffer_observed_fraction': pct(a_conc, a_zone),
            })
    coverage_df = pd.DataFrame(coverage_rows)
    sensitivity_df = pd.DataFrame(result_rows)
    primary_df = coverage_df.merge(
        sensitivity_df[np.isclose(sensitivity_df['rms_threshold'], PRIMARY_THRESHOLD)], on='zone_id')
    for zid, group in sensitivity_df.groupby('zone_id'):
        if np.any(np.diff(group.sort_values('rms_threshold')['concurrence_km2']) > 1e-8):
            raise RuntimeError(f'{zid}: threshold sensitivity should be non-increasing.')

    # Compare recomputed SAR denominator with the earlier rounded GEE results.
    # Different polygon-edge/area conventions may create small differences.
    reported_sar3 = {'R05': 4.9621, 'R06': 3.3296}
    sar_area_check_df = coverage_df[['zone_id', 'sar_class3_total_km2']].copy()
    sar_area_check_df['reported_GEE_class3_km2'] = sar_area_check_df['zone_id'].map(reported_sar3)
    sar_area_check_df['relative_difference_pct'] = (
        100 * (sar_area_check_df['sar_class3_total_km2'] - sar_area_check_df['reported_GEE_class3_km2'])
        / sar_area_check_df['reported_GEE_class3_km2'])
    sar_area_check_df['screen'] = np.where(
        sar_area_check_df['relative_difference_pct'].abs() <= 2,
        'within_2pct_screen_not_accuracy_validation', 'REVIEW_SOURCE_OR_BOUNDARY')
    if (sar_area_check_df['relative_difference_pct'].abs() > 2).any():
        warnings.append('SAR areas differ from earlier GEE values by >2%; review input version/zone boundaries.')

    # --------------------------------------------------------
    # F. Write authoritative binary M01 with explicit NoData.
    # --------------------------------------------------------
    binary = np.full(shape_hw, 255, dtype=np.uint8)
    binary[common] = (sar3[common] & (rms[common] >= PRIMARY_THRESHOLD)).astype(np.uint8)
    support = np.full(shape_hw, 255, dtype=np.uint8)
    support[inside & sar_valid] = optical_valid[inside & sar_valid].astype(np.uint8)
    profile = dict(driver='GTiff', width=ref['width'], height=ref['height'], count=1,
                   dtype='uint8', crs=ref['crs'], transform=t, nodata=255,
                   compress='deflate', tiled=True, blockxsize=256, blockysize=256)
    for filename, array, description in [
        ('M01_concurrence_RMS008_audited.tif', binary, 'SAR_Class3_AND_RMS_ge_0.08'),
        ('M02_joint_optical_support.tif', support, '0_SAR_only_1_optical_and_SAR_255_no_SAR_or_outside'),
    ]:
        with rasterio.open(out / filename, 'w', **profile) as dst:
            dst.write(array, 1)
            dst.set_band_description(1, description)
            dst.update_tags(analysis='candidate_change_not_confirmed_damage', version='1.0')
    pd.DataFrame([
        {'file': 'M01', 'value': 0, 'meaning': 'Both observed; concurrence criterion not met'},
        {'file': 'M01', 'value': 1, 'meaning': 'SAR Class3 and optical RMS >= 0.08'},
        {'file': 'M01', 'value': 255, 'meaning': 'Not jointly observed or outside buffers'},
        {'file': 'M02', 'value': 0, 'meaning': 'SAR observed; optical unavailable'},
        {'file': 'M02', 'value': 1, 'meaning': 'Both optical and SAR observed'},
        {'file': 'M02', 'value': 255, 'meaning': 'SAR unavailable or outside buffers'},
    ]).to_csv(out / 'output_raster_legend.csv', index=False)

    # --------------------------------------------------------
    # G. Optional audit of the M01 already created in QGIS.
    # It is checked, NOT used as the source for area calculations.
    # --------------------------------------------------------
    qgis_rows = []
    if 'qgis_m01' in local:
        try:
            qvalues, qvalid = matching_window(local['qgis_m01'], 1, 'qgis_m01')
            qvalid &= ~np.isin(qvalues, [255, -9999]) & (qvalues > -1e30)
            if np.any(qvalid & ~np.isin(qvalues, [0, 1])):
                raise ValueError('QGIS M01 contains valid values other than 0/1.')
            for code, zid in [(5, 'R05'), (6, 'R06')]:
                z = zones == code
                compare = z & common & qvalid
                disagree = int(np.count_nonzero(compare & (qvalues != binary)))
                missing_qgis = int(np.count_nonzero(z & common & ~qvalid))
                extra_positive = int(np.count_nonzero(z & ~common & qvalid & (qvalues == 1)))
                qgis_rows.append({
                    'zone_id': zid, 'compared_pixels': int(compare.sum()),
                    'different_binary_pixels': disagree,
                    'joint_pixels_missing_from_QGIS': missing_qgis,
                    'QGIS_positive_pixels_without_joint_support': extra_positive,
                    'status': 'MATCH' if not (disagree or missing_qgis or extra_positive) else 'REVIEW',
                })
            if any(row['status'] != 'MATCH' for row in qgis_rows):
                warnings.append('The supplied QGIS M01 differs from the audited input-based result.')
        except ValueError as exc:
            qgis_rows = [{'status': 'NOT_COMPARABLE', 'reason': str(exc)}]
            warnings.append('QGIS M01 could not be compared on the same grid; see its audit table.')
    else:
        qgis_rows = [{'status': 'NOT_PROVIDED', 'reason': 'QGIS M01 pixel identity was not checked.'}]
    qgis_check_df = pd.DataFrame(qgis_rows)

    # --------------------------------------------------------
    # H. Save reproducibility tables and checksums.
    # --------------------------------------------------------
    tables = {
        'M01_primary_R05_R06.csv': primary_df,
        'M01_threshold_sensitivity.csv': sensitivity_df,
        'M01_observation_coverage.csv': coverage_df,
        'SAR_area_consistency_check.csv': sar_area_check_df,
        'QGIS_M01_comparison.csv': qgis_check_df,
    }
    for filename, frame in tables.items():
        frame.to_csv(out / filename, index=False)
    shutil.copyfile(local['zones'], out / 'river_reference_buffers_500m_used.geojson')

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    inputs_audit = [{
        'role': role, 'source_path': str(paths[role]),
        'bytes': path.stat().st_size, 'sha256': sha256(path),
    } for role, path in local.items()]
    pd.DataFrame(inputs_audit).to_csv(out / 'input_checksums.csv', index=False)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'code_version': '1.0', 'inputs': inputs_audit, 'raster_metadata': input_metadata,
        'primary_rule': 'original SAR code == 3 AND stored spectral RMS >= 0.08',
        'additional_RMS_sensitivity_thresholds': thresholds,
        'resampling': 'None; exact integer-aligned windows only',
        'area_method': 'Pixel-center zonal counting x 400 m2 on EPSG:32645 grid',
        'geometry_edge_convention': 'R05 gets any shared boundary pixel; no polygon repair',
        'NoData_rule': 'Both inputs must be observed; missing optical is never no-concurrence',
        'interpretation': 'Cross-sensor change candidate, not damage, probability or causal proof',
        'limitations': [
            'Provisional river-reference geometry and geometrically proposed endpoints.',
            'SAR Class3 is relative to only two pre-event controls.',
            'Optical RMS is not temporally controlled by this cell.',
            'Optical compositing and SAR acquisition windows are not identical.',
            'Optical indices from the same imagery are not independent validations.',
            'UTM pixel-center areas may differ from weighted Earth Engine pixelArea sums.',
            'A lower all-Class3 fraction can reflect missing optical observations.',
        ],
        'warnings': warnings,
        'software': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                     'rasterio': rasterio.__version__, 'GDAL': rasterio.__gdal_version__,
                     'shapely': shapely.__version__, 'pyproj': pyproj.__version__},
    }
    (out / 'M01_method_and_inputs.json').write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    # Raw TIFFs are not recopied into the output bundle; hashes identify them.
    result_files = sorted(p for p in out.iterdir() if p.is_file())
    pd.DataFrame([{'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in result_files]).to_csv(out / 'output_checksums.csv', index=False)
    zip_path = out / 'Lhende_M01_quantification_bundle.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file() and path != zip_path:
                archive.write(path, arcname=path.name)
    drive_destination = None
    if save_to_drive:
        drive_destination = drive_root / 'Lhende_2026_Multisensor_Concurrence' / out.name
        drive_destination.mkdir(parents=True, exist_ok=False)
        for path in sorted(out.iterdir()):
            if path.is_file():
                target = drive_destination / path.name
                shutil.copyfile(path, target)
                if sha256(target) != sha256(path):
                    raise IOError(f'Drive checksum verification failed: {target.name}')
        print('\nSaved and checksum-verified on Drive:', drive_destination)

    print('\nOBSERVATION COVERAGE — DO NOT IGNORE OPTICAL GAPS')
    display(coverage_df.round(4))
    print('\nPRIMARY M01 CONCURRENCE — RMS >= 0.08')
    columns = ['zone_id', 'sar_class3_total_km2', 'sar_class3_optically_assessable_km2',
               'sar_class3_optical_unavailable_km2', 'concurrence_km2',
               'concurrence_pct_of_assessable_class3',
               'concurrence_pct_of_all_class3_observed_fraction']
    with pd.option_context('display.max_columns', None):
        display(primary_df[columns].round(4))
    print('\nRMS THRESHOLD SENSITIVITY')
    display(sensitivity_df[['zone_id', 'rms_threshold', 'concurrence_km2',
                            'concurrence_pct_of_assessable_class3']].round(4))
    print('\nSAR AREA CONSISTENCY SCREEN')
    display(sar_area_check_df.round(4))
    print('\nOPTIONAL QGIS M01 CHECK')
    display(qgis_check_df)
    for warning in warnings:
        print('REVIEW:', warning)
    print('\nLocal results:', out)
    print('No old files, geometries, scores or Earth Engine tasks were changed.')
    globals().update(m01_primary_df=primary_df, m01_sensitivity_df=sensitivity_df,
                     m01_coverage_df=coverage_df, m01_qgis_check_df=qgis_check_df,
                     M01_OUTPUT_DIR=out, M01_DRIVE_DIR=drive_destination)
    return primary_df


if __name__ == '__main__':
    run_m01_quantification(INPUT_FILES, SAVE_TO_DRIVE)
