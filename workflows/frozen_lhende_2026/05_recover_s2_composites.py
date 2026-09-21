# ============================================================
# LHENDE: RECOVER SAVED S2 PRE/POST COMPOSITES FOR PATCH REVIEW
# v1.0.0 -- Google Colab, file-based, NO Earth Engine requests
#
# R01 and R02 are EXPORT NUMBERS, NOT geographic reach numbers.
# Reuses retained 20 m reflectance/index exports, checks them
# against the R03 used in M01, and copies their pixel values.
# Does not recreate missing source scenes or per-pixel dates.
# ============================================================
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from contextlib import ExitStack

CONFIG = {
    'audit_dir': '/content/drive/MyDrive/Lhende_2026_Multisensor_Concurrence/M01_audit_20260911T072517612102Z',
    'export_dir': '/content/drive/MyDrive/Lhende_2026_GEE_Exports',
    # Optional overrides: use Colab/Drive paths, NOT Windows I:\\... paths.
    'pre_file': '',
    'post_file': '',
    'change_file': '',  # Must match the original M01 optical-source SHA-256.
    'save_to_drive': True,
    'allow_upload_missing_exports': True,
    'output_root': '/content/Lhende_2026_S2_Recovery',
    'drive_output_root': '/content/drive/MyDrive/Lhende_2026_S2_Review',
    'write_all_exported_band_crops': True,
    'write_ready_to_view_rgba': True,
    # Fixed MATCHED display transfer, from the original optical visualization.
    # Only the optional DISPLAY_ONLY files use this transfer.
    'display_min': 0.02,
    'display_max': 0.35,
    'display_gamma': 1.1,
    # Numeric comparison tolerance accounts for separately rounded Float32 exports.
    'absolute_tolerance': 0.000002,
    'relative_tolerance': 0.00001,
    'max_pixels': 5_000_000,
}

for package in ['numpy', 'pandas', 'rasterio', 'shapely', 'pyproj']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.enums import ColorInterp
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as transform_geometry
from pyproj import Transformer, CRS as Projection

VERSION = '1.0.0'
REFLECTANCE = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
INDICES = ['NDVI', 'NDSI', 'MNDWI', 'NDMI', 'NBR', 'BSI']
RGB = ['B4', 'B3', 'B2']
AUDIT_FILES = {
    'm01': 'M01_concurrence_RMS008_audited.tif',
    'm02': 'M02_joint_optical_support.tif',
    'zones': 'river_reference_buffers_500m_used.geojson',
    'primary': 'M01_primary_R05_R06.csv',
    'metadata': 'M01_method_and_inputs.json',
    'input_checksums': 'input_checksums.csv',
}


def check(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def display_table(frame):
    try:
        from IPython.display import display
        with pd.option_context('display.max_columns', None):
            display(frame)
    except ImportError:
        print(frame.to_string(index=False))


def clean_band_name(name):
    text = (name or '').strip()
    text = re.sub(r'^band\s*\d+\s*:\s*', '', text, flags=re.I)
    text = re.sub(r'\s*\(gray\)\s*$', '', text, flags=re.I)
    return text.casefold()


def band_map(ds, required, role):
    names = [clean_band_name(x) for x in ds.descriptions]
    result = {}
    for band in required:
        found = [i for i, name in enumerate(names) if name == band.casefold()]
        check(len(found) == 1,
              f'{role}: cannot unambiguously locate band {band}. '
              'Use the original named-band export; no band order will be guessed.')
        result[band] = found[0]  # zero-based for NumPy
    return result


def dataset_info(ds):
    return {
        'crs': str(ds.crs), 'transform': list(ds.transform)[:6],
        'width': ds.width, 'height': ds.height, 'count': ds.count,
        'descriptions': list(ds.descriptions), 'dtypes': list(ds.dtypes),
        'nodata': [None if x is None else float(x) for x in ds.nodatavals],
        'scales': list(ds.scales), 'offsets': list(ds.offsets),
        'dataset_tags': ds.tags(),
    }


def copy_raster_and_sidecars(source, destination, role, input_rows):
    """Keep sidecar masks/metadata when caching; record their hashes as well."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    pairs = [(Path(source), destination)]
    for suffix in ['.msk', '.aux.xml', '.ovr']:
        src = Path(str(source) + suffix)
        if src.is_file():
            pairs.append((src, Path(str(destination) + suffix)))
    for src, dst in pairs:
        digest = sha256(src)
        shutil.copyfile(src, dst)
        check(sha256(dst) == digest, f'Local copy failed verification: {src.name}')
        input_rows.append({'role': role, 'source_path': str(src),
                           'file': src.name, 'bytes': src.stat().st_size, 'sha256': digest})
    return destination


def read_aligned(ds, ref, role):
    """Read a full-resolution integer window. No out_shape/warp/resampling."""
    relative = (~ds.transform) * ref['transform']
    check(ds.crs == ref['crs'], f'{role}: CRS differs. No reprojection was performed.')
    check(np.allclose([relative.a, relative.b, relative.d, relative.e], [1, 0, 0, 1],
                      atol=1e-8, rtol=0) and
          np.allclose([relative.c, relative.f], np.round([relative.c, relative.f]),
                      atol=1e-6, rtol=0),
          f'{role}: grid is not integer-aligned with the audited M01 grid. Stop; do not resample to conceal this.')
    check(ds.count <= 32, f'{role}: unexpected band count.')
    check(all(dtype == 'float32' for dtype in ds.dtypes),
          f'{role}: expected the original Float32 export; source values will not be silently converted.')
    check(all(x == 1 for x in ds.scales) and all(x == 0 for x in ds.offsets),
          f'{role}: unexpected scale/offset. Do not apply an additional 0.0001 factor.')
    window = Window(int(round(relative.c)), int(round(relative.f)), ref['width'], ref['height'])
    array = ds.read(window=window, masked=True, boundless=True)
    values = np.asarray(array.data, dtype=np.float32)
    valid = ~np.ma.getmaskarray(array) & np.isfinite(values)
    for i, nodata in enumerate(ds.nodatavals):
        if nodata is not None:
            valid[i] &= values[i] != nodata
    valid &= values != -9999.0
    return values, valid


def read_audited_mask(path):
    with rasterio.open(path) as ds:
        check(ds.count == 1 and ds.crs == rasterio.crs.CRS.from_epsg(32645),
              f'{path.name}: expected one-band EPSG:32645 mask.')
        check(np.allclose([ds.transform.a, ds.transform.b, ds.transform.d, ds.transform.e],
                          [20, 0, 0, -20], atol=1e-8, rtol=0), 'Expected north-up 20 m audit grid.')
        check(ds.scales[0] == 1 and ds.offsets[0] == 0, 'Unexpected mask scale/offset.')
        a = ds.read(1, masked=True)
        values = np.asarray(a.data)
        valid = ~np.ma.getmaskarray(a) & np.isfinite(values) & (values != 255)
        check(np.all(np.isin(values[valid], [0, 1])), f'{path.name}: unexpected valid mask values.')
        ref = {'crs': ds.crs, 'transform': ds.transform, 'width': ds.width,
               'height': ds.height, 'bounds': tuple(ds.bounds)}
    return values, valid, ref


def write_float_subset(path, values, valid, descriptions, ref, period, scope):
    """Write retained Float32 values; only NoData and spatial masks are imposed."""
    check(values.shape == valid.shape, 'Invalid export mask shape.')
    array = np.where(valid, values, np.float32(-9999)).astype(np.float32, copy=False)
    profile = dict(driver='GTiff', width=ref['width'], height=ref['height'],
                   count=values.shape[0], dtype='float32', crs=ref['crs'],
                   transform=ref['transform'], nodata=-9999.0, compress='deflate',
                   predictor=3, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array)
        for i, desc in enumerate(descriptions, 1):
            if desc:
                dst.set_band_description(i, desc)
        if list(descriptions) == RGB:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        dst.update_tags(recovery_version=VERSION, period=period, scope=scope,
                        resampling='none', source_values='unchanged Float32',
                        temporal_meaning='original exported multiscene composite; not a single dated scene',
                        use='source-based interpretation, not independent impact validation')
    with rasterio.open(path) as ds:
        stored = ds.read(masked=True)
        check(ds.transform == ref['transform'] and ds.crs == ref['crs'], 'Output grid changed.')
        check(np.array_equal(~np.ma.getmaskarray(stored), valid), f'Output mask mismatch: {path.name}')
        check(np.array_equal(stored.data[valid], values[valid]), f'Pixel values changed: {path.name}')
    return {'file': path.name, 'scope': scope, 'band_count': values.shape[0],
            'value_copy_check': 'EXACT_ON_RETAINED_PIXELS',
            'valid_samples': int(valid.sum())}


def write_display_only(path, rgb_values, joint, ref, cfg):
    lo, hi, gamma = cfg['display_min'], cfg['display_max'], cfg['display_gamma']
    scaled = np.clip((rgb_values.astype(np.float64) - lo) / (hi - lo), 0, 1)
    scaled = np.power(scaled, 1.0 / gamma)
    out = np.zeros((4, ref['height'], ref['width']), dtype=np.uint8)
    out[:3] = np.where(joint[None], np.rint(255 * scaled), 0).astype(np.uint8)
    out[3] = joint.astype(np.uint8) * 255
    with rasterio.open(path, 'w', driver='GTiff', width=ref['width'], height=ref['height'],
                       count=4, dtype='uint8', crs=ref['crs'], transform=ref['transform'],
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256) as ds:
        ds.write(out)
        ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        ds.descriptions = ('B4_DISPLAY', 'B3_DISPLAY', 'B2_DISPLAY', 'alpha')
        ds.update_tags(use='DISPLAY_ONLY_NOT_REFLECTANCE', min=str(lo), max=str(hi),
                       gamma=str(gamma), transfer='round(255*clip((reflectance-min)/(max-min),0,1)^(1/gamma))',
                       missing_pixels='alpha=0', independent_evidence='no')
    with rasterio.open(path) as ds:
        check(np.array_equal(ds.read(), out), 'Display file readback failed.')


def resolve_exports(cfg, optical_source, out):
    exports = Path(cfg['export_dir'])
    defaults = {
        'pre': exports / 'R01_S2_pre_reflectance_indices_RETRY.tif',
        'post': exports / 'R02_S2_post_reflectance_indices_RETRY.tif',
        'change': Path(optical_source['source_path']),
    }
    result = {}
    for role in defaults:
        override = cfg.get(role + '_file', '')
        path = Path(override) if override else defaults[role]
        if path.is_file():
            result[role] = path
    if 'change' not in result and not cfg.get('change_file'):
        fallback = exports / 'R03_optical_change_metrics_RETRY.tif'
        if fallback.is_file():
            result['change'] = fallback
    missing = [r for r in defaults if r not in result]
    if missing and cfg['allow_upload_missing_exports']:
        print('\nMissing saved exports:', ', '.join(missing))
        print('Upload only the missing original PRE/POST/R03 TIFF files. No new scenes will be substituted.')
        from google.colab import files
        uploaded = files.upload()
        upload_dir = out / '_working' / 'uploaded'
        upload_dir.mkdir(parents=True, exist_ok=True)
        grouped = {}
        for name, data in uploaded.items():
            stem = re.sub(r'[^a-z0-9]', '', Path(name).stem.lower())
            role = None
            if stem.startswith('r01s2prereflectanceindices'):
                role = 'pre'
            elif stem.startswith('r02s2postreflectanceindices'):
                role = 'post'
            elif stem.startswith('r03opticalchangemetrics'):
                role = 'change'
            if role in missing and Path(name).suffix.lower() in {'.tif', '.tiff'}:
                path = upload_dir / Path(name).name
                path.write_bytes(data)
                grouped.setdefault(role, []).append(path)
        for role, paths in grouped.items():
            check(len(paths) == 1, f'Multiple uploaded {role} exports. Do not choose a version arbitrarily.')
            result[role] = paths[0]
    missing = [r for r in defaults if r not in result]
    if missing:
        expected = '\n'.join(f'{role}: {defaults[role]}' for role in missing)
        raise FileNotFoundError('Missing retained exports. No exact scene reconstruction was attempted.\n' + expected)
    check(len({str(p.resolve()) for p in result.values()}) == 3, 'PRE, POST and R03 must be different files.')
    check(sha256(result['change']) == optical_source['sha256'],
          'R03 does not match the optical source used in the successful M01 audit. Do not mix runs.')
    return result


def run_s2_recovery(config=None, source_code_path=None):
    cfg = {**CONFIG, **(config or {})}
    check(cfg['display_max'] > cfg['display_min'] and cfg['display_gamma'] > 0,
          'Invalid display settings.')
    check(cfg['absolute_tolerance'] >= 0 and cfg['relative_tolerance'] >= 0, 'Invalid numeric tolerance.')
    drive_root = Path('/content/drive/MyDrive')
    if (cfg['save_to_drive'] or str(cfg['audit_dir']).startswith('/content/drive/')) and not drive_root.is_dir():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    audit = Path(cfg['audit_dir'])
    check(audit.is_dir(), f'Audited source folder not found: {audit}')
    checksum_path = audit / 'output_checksums.csv'
    check(checksum_path.is_file(), 'Missing M01 audit output_checksums.csv.')
    checksums = pd.read_csv(checksum_path, keep_default_na=False)
    check({'file', 'bytes', 'sha256'} <= set(checksums), 'Invalid M01 checksum manifest.')
    check(not checksums['file'].duplicated().any(), 'Duplicate M01 manifest entries.')
    checksums = checksums.set_index('file')
    for name in AUDIT_FILES.values():
        p = audit / name
        check(p.is_file() and name in checksums.index, f'Missing audited source: {name}')
        check(p.stat().st_size == int(checksums.loc[name, 'bytes']) and
              sha256(p) == str(checksums.loc[name, 'sha256']), f'Changed audited source: {name}')
    method = json.loads((audit / AUDIT_FILES['metadata']).read_text())
    optical_sources = [x for x in method['inputs'] if x['role'] == 'optical']
    check(len(optical_sources) == 1, 'Ambiguous optical-source provenance in M01.')
    optical_source = optical_sources[0]
    # This run used the original 14-band R03 file, as recorded by the prior audit.
    check(method['raster_metadata']['optical']['band_count'] == 14,
          'This audit used only an extracted RMS band. Obtain and independently link the full original R03 first.')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(cfg['output_root']) / ('S2_export_recovery_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    input_rows, export_rows, check_rows, warnings = [], [], [], []
    destination = None

    try:
        paths = resolve_exports(cfg, optical_source, out)
        print('\nSAVED EXPORTS USED — NO NEW SATELLITE SEARCH')
        for role, p in paths.items():
            print(f'{role.upper()}: {p}')
        working = out / '_working'
        local = {}
        for role, p in paths.items():
            local[role] = copy_raster_and_sidecars(p, working / role / p.name, role, input_rows)
        pd.DataFrame(input_rows).to_csv(out / 'source_export_checksums.csv', index=False)
        for name in AUDIT_FILES.values():
            shutil.copyfile(audit / name, working / name)
        shutil.copyfile(checksum_path, working / 'upstream_output_checksums.csv')

        m01, valid01, ref = read_audited_mask(working / AUDIT_FILES['m01'])
        m02, valid02, ref02 = read_audited_mask(working / AUDIT_FILES['m02'])
        check(ref == ref02, 'M01/M02 grid mismatch.')
        check(ref['width'] * ref['height'] <= cfg['max_pixels'], 'Unexpectedly large review window.')
        check(np.array_equal(valid01, valid02 & (m02 == 1)), 'M01/M02 support mismatch.')
        geo = json.loads((working / AUDIT_FILES['zones']).read_text(encoding='utf-8-sig'))
        check(geo.get('type') == 'FeatureCollection' and len(geo.get('features', [])) == 2,
              'Expected the two audited buffer polygons.')
        zcrs = Projection.from_user_input(geo.get('crs', {}).get('properties', {}).get('name') or 'EPSG:4326')
        project = Transformer.from_crs(zcrs, ref['crs'], always_xy=True)
        geometries = {}
        for f in geo['features']:
            p = f['properties']; zid = p.get('zone_id')
            check(zid in {'R05', 'R06'} and zid not in geometries and int(p.get('half_width_m', -1)) == 500
                  and p.get('geometry_version') == 'river_reference_candidate_v1', 'Wrong geometry version.')
            g = shape(f['geometry'])
            check(g.geom_type in {'Polygon', 'MultiPolygon'} and g.is_valid and not g.is_empty, 'Invalid source polygon.')
            g = transform_geometry(project.transform, g)
            check(g.is_valid and g.difference(box(*ref['bounds'])).area <= 400,
                  'Audited grid does not contain the expected river buffer.')
            geometries[zid] = g
        check(geometries['R05'].intersection(geometries['R06']).area <= 1.0, 'Zone overlap detected.')
        zones = np.zeros(m01.shape, dtype=np.uint8)
        for code, zid in [(5, 'R05'), (6, 'R06')]:
            z = rasterize([(mapping(geometries[zid]), 1)], out_shape=m01.shape,
                          transform=ref['transform'], all_touched=False, dtype='uint8') > 0
            zones[z & (zones == 0)] = code
        inside = zones > 0
        check(not np.any((valid01 | valid02) & ~inside), 'Audited data outside reconstructed zones.')
        primary = pd.read_csv(working / AUDIT_FILES['primary']).set_index('zone_id')
        for code, zid in [(5, 'R05'), (6, 'R06')]:
            check(int(np.count_nonzero((zones == code) & valid01 & (m01 == 1))) ==
                  int(primary.loc[zid, 'concurrence_pixel_count']), f'{zid}: M01 pixel-count mismatch.')

        with ExitStack() as stack:
            datasets = {r: stack.enter_context(rasterio.open(p)) for r, p in local.items()}
            metadata = {r: dataset_info(ds) for r, ds in datasets.items()}
            maps = {
                'pre': band_map(datasets['pre'], REFLECTANCE + INDICES, 'PRE'),
                'post': band_map(datasets['post'], REFLECTANCE + INDICES, 'POST'),
                'change': band_map(datasets['change'], ['d' + b for b in REFLECTANCE + INDICES] + ['spectral_rms'], 'R03'),
            }
            pre, vp = read_aligned(datasets['pre'], ref, 'PRE')
            post, vq = read_aligned(datasets['post'], ref, 'POST')
            r03, vc = read_aligned(datasets['change'], ref, 'R03')
            desc_pre = list(datasets['pre'].descriptions)
            desc_post = list(datasets['post'].descriptions)
        print('\nGRID CHECK PASS: original 20 m pixel lattice; no resampling.')

        # ----------------------------------------------------
        # Numeric consistency with the verified original R03.
        # Source formula clamps reflectance at zero FOR CHANGE
        # CALCULATION ONLY. Exported PRE/POST RGB values remain
        # unchanged, including any valid negative reflectance.
        # ----------------------------------------------------
        squared_sum = np.zeros(m01.shape, dtype=np.float64)
        paired_spectral_count = np.zeros(m01.shape, dtype=np.uint8)
        domains = [('R05', zones == 5), ('R06', zones == 6)]

        def compare_metric(name, calculated, available, reference_index):
            for zid, zmask in domains:
                target_valid = zmask & vc[reference_index]
                missing_values = int(np.count_nonzero(target_valid & ~available))
                take = target_valid & available
                count = int(take.sum())
                difference = np.abs(calculated[take] - r03[reference_index][take].astype(np.float64))
                allowable = cfg['absolute_tolerance'] + cfg['relative_tolerance'] * np.abs(r03[reference_index][take])
                bad = int(np.count_nonzero(difference > allowable))
                check_rows.append({
                    'zone_id': zid, 'metric': name, 'compared_pixels': count,
                    'R03_valid_pixels_without_source_inputs': missing_values,
                    'pixels_outside_numeric_tolerance': bad,
                    'maximum_absolute_difference': float(difference.max()) if count else None,
                    'status': 'PASS' if count > 0 and missing_values == 0 and bad == 0 else 'FAIL',
                })

        for band in REFLECTANCE:
            i, j = maps['pre'][band], maps['post'][band]
            available = vp[i] & vq[j]
            delta = np.maximum(post[j].astype(np.float64), 0) - np.maximum(pre[i].astype(np.float64), 0)
            compare_metric('d' + band, delta, available, maps['change']['d' + band])
            squared_sum += np.where(available, delta ** 2, 0)
            paired_spectral_count += available.astype(np.uint8)
        calculated_rms = np.sqrt(np.divide(squared_sum, paired_spectral_count,
                                          out=np.zeros_like(squared_sum), where=paired_spectral_count > 0))
        compare_metric('spectral_rms', calculated_rms, paired_spectral_count > 0,
                       maps['change']['spectral_rms'])
        for name in INDICES:
            i, j = maps['pre'][name], maps['post'][name]
            delta = post[j].astype(np.float64) - pre[i].astype(np.float64)
            compare_metric('d' + name, delta, vp[i] & vq[j], maps['change']['d' + name])

        consistency = pd.DataFrame(check_rows)
        consistency.to_csv(out / 'PRE_POST_vs_R03_consistency.csv', index=False)
        if not consistency['status'].eq('PASS').all():
            display_table(consistency)
            raise ValueError('PRE/POST do not reproduce R03 within the stated tolerance. '
                             'No review rasters were accepted. Check export versions; do not loosen the tolerance blindly.')
        rms_available = vc[maps['change']['spectral_rms']]
        check(np.array_equal(valid01, valid02 & rms_available), 'R03 availability does not reproduce audited joint support.')
        print('NUMERIC CONSISTENCY PASS: six reflectance deltas, six index deltas and spectral RMS.')

        rgb_pre = pre[[maps['pre'][b] for b in RGB]]
        rgb_post = post[[maps['post'][b] for b in RGB]]
        rgb_pre_valid = np.all(vp[[maps['pre'][b] for b in RGB]], axis=0) & inside
        rgb_post_valid = np.all(vq[[maps['post'][b] for b in RGB]], axis=0) & inside
        joint = valid01 & rgb_pre_valid & rgb_post_valid
        check(joint.any(), 'No matched RGB review pixels.')

        coverage_rows = []
        pixel_km2 = 0.0004
        for code, zid in [(5, 'R05'), (6, 'R06')]:
            z = zones == code
            pixels = int(z.sum())
            target = z & valid01 & (m01 == 1)
            missing_rgb = int(np.count_nonzero(target & ~joint))
            if missing_rgb:
                warnings.append(f'{zid}: {missing_rgb} M01 pixels do not have all three RGB bands for both dates.')
            coverage_rows.append({
                'zone_id': zid, 'buffer_grid_area_km2': pixels * pixel_km2,
                'PRE_RGB_available_km2': int(np.count_nonzero(z & rgb_pre_valid)) * pixel_km2,
                'POST_RGB_available_km2': int(np.count_nonzero(z & rgb_post_valid)) * pixel_km2,
                'audited_joint_optical_km2': int(np.count_nonzero(z & valid01)) * pixel_km2,
                'matched_RGB_review_km2': int(np.count_nonzero(z & joint)) * pixel_km2,
                'matched_RGB_review_pct_of_buffer': 100.0 * np.count_nonzero(z & joint) / pixels,
                'M01_candidate_pixels': int(target.sum()),
                'M01_candidates_without_matched_RGB': missing_rgb,
            })
        coverage = pd.DataFrame(coverage_rows)
        coverage.to_csv(out / 'RGB_review_coverage.csv', index=False)
        del r03, vc, squared_sum, calculated_rms

        # ----------------------------------------------------
        # Exact Float32 RGB copies on identical paired support.
        # ----------------------------------------------------
        rgb_mask = np.broadcast_to(joint, rgb_pre.shape)
        export_rows.append(write_float_subset(out / 'V01_R05_R06_S2_PRE_RGB.tif', rgb_pre, rgb_mask,
                                             RGB, ref, 'PRE', 'matched_audited_joint_support'))
        export_rows.append(write_float_subset(out / 'V02_R05_R06_S2_POST_RGB.tif', rgb_post, rgb_mask,
                                             RGB, ref, 'POST', 'matched_audited_joint_support'))
        if cfg['write_ready_to_view_rgba']:
            write_display_only(out / 'V01_PRE_RGB_DISPLAY_ONLY.tif', rgb_pre, joint, ref, cfg)
            write_display_only(out / 'V02_POST_RGB_DISPLAY_ONLY.tif', rgb_post, joint, ref, cfg)
        if cfg['write_all_exported_band_crops']:
            export_rows.append(write_float_subset(out / 'V04_PRE_original_export_bands_R05_R06.tif',
                                                 pre, vp & inside[None], desc_pre, ref, 'PRE',
                                                 'per_band_source_masks_cropped_to_audited_buffers'))
            export_rows.append(write_float_subset(out / 'V05_POST_original_export_bands_R05_R06.tif',
                                                 post, vq & inside[None], desc_post, ref, 'POST',
                                                 'per_band_source_masks_cropped_to_audited_buffers'))
        pd.DataFrame(export_rows).to_csv(out / 'export_pixel_identity_checks.csv', index=False)

        support = np.full(m01.shape, 255, dtype=np.uint8)
        support[inside] = 0
        support[rgb_pre_valid & ~rgb_post_valid] = 1
        support[rgb_post_valid & ~rgb_pre_valid] = 2
        support[rgb_pre_valid & rgb_post_valid] = 3
        support[joint] = 4
        with rasterio.open(out / 'V03_RGB_review_support.tif', 'w', driver='GTiff', width=ref['width'],
                           height=ref['height'], count=1, dtype='uint8', crs=ref['crs'],
                           transform=ref['transform'], nodata=255, compress='deflate',
                           tiled=True, blockxsize=256, blockysize=256) as ds:
            ds.write(support, 1)
            ds.set_band_description(1, 'RGB_review_availability_NOT_change')
        with rasterio.open(out / 'V03_RGB_review_support.tif') as ds:
            check(np.array_equal(ds.read(1), support), 'Support TIFF readback failure.')
        pd.DataFrame([
            (0, 'Neither PRE nor POST has all RGB bands'),
            (1, 'Only PRE has all RGB bands'), (2, 'Only POST has all RGB bands'),
            (3, 'Both RGB available but excluded from matched audited review support'),
            (4, 'Both RGB available AND audited joint optical/SAR support'),
            (255, 'Outside R05/R06 buffers'),
        ], columns=['value', 'meaning']).to_csv(out / 'V03_support_legend.csv', index=False)
        shutil.copyfile(working / AUDIT_FILES['zones'], out / AUDIT_FILES['zones'])
        shutil.copyfile(working / AUDIT_FILES['m02'], out / AUDIT_FILES['m02'])

        run_metadata = {
            'created_utc': datetime.now(timezone.utc).isoformat(), 'code_version': VERSION,
            'source_M01_audit': str(audit), 'settings': cfg, 'source_exports': input_rows,
            'source_dataset_metadata': metadata, 'band_maps_zero_based': maps,
            'output_grid': {'crs': str(ref['crs']), 'transform': list(ref['transform'])[:6],
                            'width': ref['width'], 'height': ref['height']},
            'recovery_scope': 'Subset of already exported 20 m composites. Not a re-query of Sentinel-2.',
            'pixel_identity': 'Float32 values copied exactly on retained support; masks documented per output.',
            'consistency_check': 'Signed deltas and RMS reproduce hashed R03 within numeric tolerance in the audited buffers.',
            'change_reflectance_formula': 'max(POST_band,0)-max(PRE_band,0); RMS=sqrt(mean(delta^2))',
            'index_delta_formula': 'stored_POST_index - stored_PRE_index',
            'rgb_band_order': ['B4', 'B3', 'B2'],
            'limitations': [
                'Consistency of differences cannot uniquely prove the historical source pair or recover discarded provenance.',
                'PRE/POST source hashes are recorded now; no older PRE/POST hashes were available in the M01 audit.',
                'These are retained 20 m exports; no native-resolution 10 m image is recreated.',
                'Per-pixel acquisition dates, scene IDs and solar metadata not retained in the source exports are not recoverable here.',
                'Original compositing/cloud/illumination artefacts are not corrected by extracting RGB.',
                'Matched RGB outputs deliberately share a narrower joint mask; per-date band crops keep original band masks inside the buffers.',
                'The same optical imagery used to detect M01 is not independent event validation.',
                'The river-reference geometry and named endpoints remain provisional.',
            ],
            'warnings': warnings,
            'software': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                         'rasterio': rasterio.__version__, 'GDAL': rasterio.__gdal_version__},
        }
        write_json(out / 'S2_recovery_manifest.json', run_metadata)
        instructions = f'''SENTINEL-2 EXPORTED-COMPOSITE RECOVERY v{VERSION}

IMPORTANT CORRECTION
R01_S2_pre_reflectance_indices_RETRY is the PRE composite export.
R02_S2_post_reflectance_indices_RETRY is the POST composite export.
R01/R02 here are output numbers, NOT geographic reach identifiers.

NO NEW GEE JOBS, SCENE SEARCH, RESAMPLING, GAP FILLING OR REMASKING OF SOURCE FILES.
Inputs and old analysis files are unchanged.

OUTPUTS
V01_R05_R06_S2_PRE_RGB.tif / V02_R05_R06_S2_POST_RGB.tif:
  Original Float32 B4/B3/B2 samples on identical audited RGB review support.
  NoData=-9999. No intensity scaling was applied to these quantitative rasters.
V01_PRE_RGB_DISPLAY_ONLY.tif / V02_POST_RGB_DISPLAY_ONLY.tif (if requested):
  Byte RGBA copies for easy QGIS display. Same {cfg['display_min']}..{cfg['display_max']}
  reflectance limits and gamma {cfg['display_gamma']} on both. Real alpha channel is band 4.
  DISPLAY ONLY: do not compute indices, change metrics or areas from rendered values.
V04/V05 original-export-band crops (if requested):
  All source bands copied unchanged with their original band masks, within R05/R06.
  More per-date context may be present than in the matched RGB views.
V03_RGB_review_support.tif: availability codes; see V03_support_legend.csv.

QGIS
1. Keep the patch inventory above the imagery, outline-only (No Brush).
2. Turn OFF the orange audited-buffer fill and the M01/M02/score raster fills.
3. Add the two DISPLAY_ONLY TIFFs for the easiest matched RGB comparison.
4. Toggle PRE and POST at exactly the same map extent. Do not stretch each separately.
5. For raw float RGBs use Red=Band1, Green=Band2, Blue=Band3;
   identical min={cfg['display_min']} max={cfg['display_max']} gamma={cfg['display_gamma']} on both.
6. NoData is not no-change. Current OpenStreetMap is not dated event imagery.
7. Seeing the same optical evidence that generated M01 is interpretation/QC,
   not independent validation or proof of flood/debris damage.

PROVENANCE
R03 file hash matches the successful M01 audit. Export deltas/RMS are checked
within Float32 tolerance. These tests check numerical consistency, not unique
acquisition history. Per-pixel date/scene maps are not invented.
'''
        (out / 'README_QGIS_REVIEW.txt').write_text(instructions, encoding='utf-8')
        code_snapshot = out / 'recovery_code_used.py'
        if source_code_path and Path(source_code_path).is_file():
            shutil.copyfile(source_code_path, code_snapshot)
        else:
            # In a full-code notebook cell, recover the exact executed cell.
            try:
                from IPython import get_ipython
                history = get_ipython().history_manager.input_hist_raw
                matching_cells = [cell for cell in history if
                                  'def run_s2_recovery(' in cell and
                                  'LHENDE: RECOVER SAVED S2 PRE/POST COMPOSITES' in cell]
                if matching_cells:
                    code_snapshot.write_text(matching_cells[-1], encoding='utf-8')
            except Exception:
                pass
            if not code_snapshot.is_file():
                warnings.append('An execution code snapshot was not available; retain the supplied .py/.ipynb.')
        if code_snapshot.is_file():
            run_metadata['code_snapshot_sha256'] = sha256(code_snapshot)
        write_json(out / 'S2_recovery_manifest.json', run_metadata)
        outputs = [p for p in sorted(out.iterdir()) if p.is_file()]
        pd.DataFrame([{'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in outputs]).to_csv(
            out / 'output_checksums.csv', index=False)
        with zipfile.ZipFile(out / 'Lhende_S2_export_recovery_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.iterdir()):
                if p.is_file() and p.suffix != '.zip':
                    z.write(p, arcname=p.name)
        if cfg['save_to_drive']:
            destination = Path(cfg['drive_output_root']) / out.name
            destination.mkdir(parents=True, exist_ok=False)
            for p in sorted(out.iterdir()):
                if p.is_file():
                    target = destination / p.name
                    shutil.copyfile(p, target)
                    check(sha256(target) == sha256(p), f'Drive checksum mismatch: {p.name}')
            print('\nSAVED AND CHECKSUM-VERIFIED ON DRIVE:', destination)
        print('\nMATCHED RGB REVIEW COVERAGE')
        display_table(coverage.round(4))
        print('\nEXPORTED PIXEL IDENTITY CHECKS')
        display_table(pd.DataFrame(export_rows))
        print('\nRecovery completed. No Earth Engine tasks were started.')
        print('PRE and POST are saved composite subsets, not new individual dated scenes.')
        print('Local output:', out)
        return {'local_dir': out, 'drive_dir': destination, 'consistency': consistency,
                'coverage': coverage, 'pixel_checks': pd.DataFrame(export_rows)}
    except Exception as exc:
        if check_rows:
            pd.DataFrame(check_rows).to_csv(out / 'PRE_POST_vs_R03_consistency.csv', index=False)
        write_json(out / 'RECOVERY_FAILED.json', {'error_type': type(exc).__name__, 'error': str(exc),
                                               'status': 'NOT_ACCEPTED_FOR_REVIEW',
                                               'source_files_changed': False})
        # Preserve failure diagnostics; do not call any partial raster a successful recovery.
        if cfg['save_to_drive'] and drive_root.is_dir():
            fail_dir = Path(cfg['drive_output_root']) / (out.name + '_FAILED')
            fail_dir.mkdir(parents=True, exist_ok=False)
            for name in ['RECOVERY_FAILED.json', 'PRE_POST_vs_R03_consistency.csv', 'source_export_checksums.csv']:
                p = out / name
                if p.is_file():
                    shutil.copyfile(p, fail_dir / name)
            print('Failure diagnostics saved to:', fail_dir)
        raise


if __name__ == '__main__':
    _code_path = globals().get('S2_RECOVERY_SCRIPT_PATH') or globals().get('__file__')
    S2_RECOVERY_RESULT = run_s2_recovery(CONFIG, source_code_path=_code_path)
