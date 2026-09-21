# ============================================================
# LHENDE: AUDITED PATCH DIRECTIONAL CHARACTERIZATION v1.0.0
# Full Google Colab cell / standalone Python script.
# File-based; no Earth Engine queries, no new satellite scenes.
# The same optical imagery characterizes detection; it is NOT
# independent validation of physical change or event damage.
# ============================================================
from pathlib import Path
from datetime import datetime, timezone
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import zipfile

PATCH_DIRECTIONAL_CONFIG = {
    'patch_id': 'R05_P0001',
    'patch_dir': '/content/drive/MyDrive/Lhende_2026_Candidate_Patches/M01_patch_inventory_20260911T082650777727Z',
    's2_dir': '/content/drive/MyDrive/Lhende_2026_S2_Review/S2_export_recovery_20260911T093217241209Z',
    # Optional relocated ORIGINAL R03; its hash must match the recovery/audit.
    'change_file': '',
    'export_dir': '/content/drive/MyDrive/Lhende_2026_GEE_Exports',
    'save_to_drive': True,
    'output_root': '/content/Lhende_2026_Patch_Directional',
    'drive_output_root': '/content/drive/MyDrive/Lhende_2026_Patch_Directional',
    # NEW descriptive reference-window settings, NOT damage thresholds.
    # Euclidean distance to the closest TARGET PIXEL CENTRE.
    'reference_inner_m': 40.0,
    'reference_outer_m': [100.0, 250.0, 500.0],
    'primary_reference_outer_m': 250.0,
    'reference_small_n_warning': 30,
    # Same small numerical tolerances as the verified S2 recovery.
    'absolute_tolerance': 0.000002,
    'relative_tolerance': 0.00001,
    'max_grid_pixels': 10_000_000,
    'write_pixel_table': True,
}

for _package in ['numpy', 'pandas', 'rasterio', 'scipy', 'shapely', 'pyproj']:
    if importlib.util.find_spec(_package) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _package])

import numpy as np
import pandas as pd
import rasterio
import scipy
from scipy import ndimage as ndi
from rasterio.features import rasterize, shapes
from rasterio.windows import Window, transform as window_transform
import shapely
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as transform_geometry, unary_union
import pyproj
from pyproj import Transformer, CRS as Projection

VERSION = '1.0.0'
REFLECTANCE = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
INDICES = ['NDVI', 'NDSI', 'MNDWI', 'NDMI', 'NBR', 'BSI']
MAIN_INDICES = ['NDVI', 'MNDWI', 'NBR']
# All intervals use upper-inclusive boundaries, including both infinite tails.
CLASS_DEFINITIONS = {
    'dNDVI': ([-0.30, -0.15, 0.15, 0.30], [
        'strong_NDVI_decrease', 'NDVI_decrease', 'small_change_display_interval',
        'NDVI_increase', 'strong_NDVI_increase']),
    'dMNDWI': ([-0.20, -0.05, 0.05, 0.20], [
        'strong_MNDWI_decrease', 'MNDWI_decrease', 'small_change_display_interval',
        'MNDWI_increase', 'strong_MNDWI_increase']),
    'dNBR': ([-0.25, -0.10, 0.10, 0.20], [
        'strong_negative_NBR_change', 'negative_NBR_change', 'small_change_display_interval',
        'positive_NBR_change', 'strong_positive_NBR_change']),
}
THRESHOLD_TESTS = [
    ('dNDVI', 'NDVI_decline_lt_minus_0p15', '<', -0.15),
    ('dMNDWI', 'MNDWI_decline_lt_minus_0p05', '<', -0.05),
    ('dMNDWI', 'MNDWI_increase_gt_0p05', '>', 0.05),
    ('dNBR', 'NBR_decline_lt_minus_0p10', '<', -0.10),
    ('spectral_rms', 'RMS_ge_0p08', '>=', 0.08),
]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def show(frame):
    try:
        from IPython.display import display
        with pd.option_context('display.max_columns', None):
            display(frame)
    except ImportError:
        print(frame.to_string(index=False))


def integer(value, label):
    value = float(value)
    require(np.isfinite(value) and value == round(value), f'{label} is not an integer.')
    return int(value)


def read_manifest(root):
    path = Path(root) / 'output_checksums.csv'
    require(path.is_file(), f'Missing original checksum manifest: {path}')
    table = pd.read_csv(path, keep_default_na=False)
    require({'file', 'bytes', 'sha256'} <= set(table.columns), f'Wrong checksum schema: {path}')
    require(not table['file'].duplicated().any(), f'Duplicate checksum entries: {path}')
    return table.set_index('file')


def cache_verified(root, manifest, name, destination, source_rows, role):
    src = Path(root) / name
    require(name in manifest.index and src.is_file(), f'Missing audited input: {src}')
    row = manifest.loc[name]
    require(src.stat().st_size == int(row['bytes']), f'Input size changed: {src}')
    expected = str(row['sha256']).lower()
    require(sha256(src) == expected, f'Input checksum changed: {src}')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, destination)
    require(sha256(destination) == expected, f'Local copy mismatch: {src}')
    source_rows.append({'role': role, 'source_path': str(src), 'bytes': src.stat().st_size,
                        'sha256': expected})
    # A new external mask must not change a previously hashed audit silently.
    if src.suffix.lower() in {'.tif', '.tiff'}:
        for suffix in ['.msk', '.aux.xml', '.ovr']:
            extra = Path(str(src) + suffix)
            if extra.is_file():
                extra_name = name + suffix
                require(extra_name in manifest.index,
                        f'Unmanifested raster sidecar found: {extra}. Use the immutable original output.')
                cache_verified(root, manifest, extra_name, Path(str(destination) + suffix),
                               source_rows, role + suffix)
    return destination


def read_mask(path):
    with rasterio.open(path) as ds:
        require(ds.count == 1 and str(ds.crs) == 'EPSG:32645', f'{path}: wrong mask CRS/bands.')
        t = ds.transform
        require(np.allclose([t.a, t.b, t.d, t.e], [20, 0, 0, -20], rtol=0, atol=1e-8),
                f'{path}: expected north-up 20 m grid.')
        require(ds.scales[0] == 1 and ds.offsets[0] == 0, f'{path}: scaled mask.')
        array = ds.read(1, masked=True)
        values = np.asarray(array.data)
        valid = ~np.ma.getmaskarray(array) & np.isfinite(values) & (values != 255)
        require(np.isin(values[valid], [0, 1]).all(), f'{path}: invalid mask values.')
        ref = {'crs': str(ds.crs), 'transform': t, 'width': ds.width, 'height': ds.height}
    return values, valid, ref


def read_zones(path, ref):
    obj = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(obj.get('type') == 'FeatureCollection' and len(obj.get('features', [])) == 2,
            'Expected the two audited R05/R06 buffers.')
    source_crs = obj.get('crs', {}).get('properties', {}).get('name') or 'EPSG:4326'
    converter = Transformer.from_crs(Projection.from_user_input(source_crs), 32645, always_xy=True)
    geometries = {}
    for item in obj['features']:
        p = item['properties']; zid = p.get('zone_id')
        require(zid in {'R05', 'R06'} and zid not in geometries and int(p.get('half_width_m', -1)) == 500
                and p.get('geometry_version') == 'river_reference_candidate_v1', 'Incorrect zone/version.')
        g = shape(item['geometry'])
        require(g.is_valid and not g.is_empty and g.geom_type in {'Polygon', 'MultiPolygon'},
                f'{zid}: invalid source polygon; no repair applied.')
        geometries[zid] = transform_geometry(converter.transform, g)
    require(geometries['R05'].intersection(geometries['R06']).area <= 1.0, 'Audited zones overlap.')
    result = np.zeros((ref['height'], ref['width']), dtype=np.uint8)
    for code, zid in [(5, 'R05'), (6, 'R06')]:
        z = rasterize([(mapping(geometries[zid]), 1)], out_shape=result.shape,
                      transform=ref['transform'], all_touched=False, fill=0, dtype='uint8') > 0
        result[z & (result == 0)] = code
    return result, geometries


def clean_band_name(value):
    value = re.sub(r'^band\s*\d+\s*:\s*', '', value or '', flags=re.I)
    return re.sub(r'\s*\(gray\)\s*$', '', value, flags=re.I).strip().casefold()


def read_named_window(path, needed, ref, role):
    with rasterio.open(path) as ds:
        require(str(ds.crs) == ref['crs'], f'{role}: CRS mismatch; no reprojection performed.')
        relative = (~ds.transform) * ref['transform']
        require(np.allclose([relative.a, relative.b, relative.d, relative.e], [1, 0, 0, 1], rtol=0, atol=1e-8)
                and np.allclose([relative.c, relative.f], np.round([relative.c, relative.f]), rtol=0, atol=1e-6),
                f'{role}: shifted pixel grid. No resampling was performed.')
        names = [clean_band_name(name) for name in ds.descriptions]
        selected = []
        for name in needed:
            matches = [i + 1 for i, band in enumerate(names) if band == name.casefold()]
            require(len(matches) == 1, f'{role}: cannot verify band {name}; do not guess its position.')
            selected.append(matches[0])
        require(all(ds.dtypes[i - 1] == 'float32' for i in selected),
                f'{role}: expected Float32 quantitative bands, not DISPLAY_ONLY pixels.')
        require(all(ds.scales[i - 1] == 1 and ds.offsets[i - 1] == 0 for i in selected),
                f'{role}: unexpected scale/offset; do not divide these exports by 10000.')
        window = Window(int(round(relative.c)), int(round(relative.f)), ref['width'], ref['height'])
        a = ds.read(indexes=selected, window=window, masked=True, boundless=True)
        data = np.asarray(a.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(a) & np.isfinite(data) & (data != -9999)
        for j, i in enumerate(selected):
            nodata = ds.nodatavals[i - 1]
            if nodata is not None:
                valid[j] &= data[j] != nodata
        return ({name: data[i] for i, name in enumerate(needed)},
                {name: valid[i] for i, name in enumerate(needed)})


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=np.nan, median=np.nan, q25=np.nan, q75=np.nan,
                    iqr=np.nan, minimum=np.nan, maximum=np.nan, sd_population=np.nan)
    q1, median, q3 = np.quantile(a, [0.25, 0.50, 0.75], method='linear')
    return dict(n=len(a), mean=float(a.mean()), median=float(median), q25=float(q1), q75=float(q3),
                iqr=float(q3 - q1), minimum=float(a.min()), maximum=float(a.max()),
                sd_population=float(a.std(ddof=0)))


def threshold_mask(values, operator, threshold):
    if operator == '<': return values < threshold
    if operator == '>': return values > threshold
    if operator == '>=': return values >= threshold
    raise ValueError(f'Unexpected comparison: {operator}')


def proportions(values, metric):
    bounds, labels = CLASS_DEFINITIONS[metric]
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    # side='left': class upper bounds are inclusive.
    codes = np.searchsorted(bounds, a, side='left')
    counts = np.bincount(codes, minlength=5)
    limits = [-np.inf, *bounds, np.inf]
    rows = []
    for i, (label, count) in enumerate(zip(labels, counts)):
        lo, hi = limits[i], limits[i + 1]
        interval = (f'x <= {hi:g}' if i == 0 else
                    f'x > {lo:g}' if i == 4 else f'{lo:g} < x <= {hi:g}')
        rows.append({'metric': metric, 'class_code': i + 1, 'class_label': label,
                     'interval': interval, 'pixels': int(count),
                     'pct_of_metric_valid_pixels': 100.0 * count / len(a) if len(a) else np.nan})
    require(int(counts.sum()) == len(a), 'Directional class partition failed.')
    return rows


def run_patch_directional(config=None, source_code_path=None):
    cfg = {**PATCH_DIRECTIONAL_CONFIG, **(config or {})}
    patch_id = str(cfg['patch_id'])
    require(re.fullmatch(r'R0[56]_P\d{4,}', patch_id) is not None, 'Use an audited primary patch_id.')
    inner = float(cfg['reference_inner_m'])
    outers = sorted(set(map(float, cfg['reference_outer_m'])))
    primary_outer = float(cfg['primary_reference_outer_m'])
    require(np.isfinite([inner, primary_outer, *outers]).all() and inner >= 0 and outers
            and min(outers) > inner and primary_outer in outers, 'Invalid local-reference distances.')
    require(cfg['absolute_tolerance'] >= 0 and cfg['relative_tolerance'] >= 0, 'Invalid tolerances.')
    patch_dir, s2_dir = Path(cfg['patch_dir']), Path(cfg['s2_dir'])
    if (cfg['save_to_drive'] or str(patch_dir).startswith('/content/drive/')
            or str(s2_dir).startswith('/content/drive/')) and not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    require(patch_dir.is_dir(), f'Original patch folder not found: {patch_dir}')
    require(s2_dir.is_dir(), f'Original S2 recovery folder not found: {s2_dir}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(cfg['output_root']) / f'{patch_id}_directional_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    work = out / '_working'
    warnings, sources = [], []

    try:
        pm, sm = read_manifest(patch_dir), read_manifest(s2_dir)
        patch_files = {
            'labels': 'patch_ids_4n.tif',
            'inventory': 'P01_candidate_patch_inventory_4n.csv',
            'm01': 'M01_concurrence_RMS008_audited.tif',
            'm02': 'M02_joint_optical_support.tif',
            'zones': 'river_reference_buffers_500m_used.geojson',
            'primary': 'source_inputs/M01_primary_R05_R06.csv',
            'method': 'source_inputs/M01_method_and_inputs.json',
            'patch_metadata': 'patch_inventory_metadata.json',
        }
        p = {role: cache_verified(patch_dir, pm, name, work / 'patch' / name, sources, role)
             for role, name in patch_files.items()}
        s2_files = {
            'pre': 'V04_PRE_original_export_bands_R05_R06.tif',
            'post': 'V05_POST_original_export_bands_R05_R06.tif',
            'recovery_metadata': 'S2_recovery_manifest.json',
            'source_records': 'source_export_checksums.csv',
            'recovery_consistency': 'PRE_POST_vs_R03_consistency.csv',
            'recovery_zones': 'river_reference_buffers_500m_used.geojson',
            'recovery_m02': 'M02_joint_optical_support.tif',
        }
        q = {role: cache_verified(s2_dir, sm, name, work / 's2' / name, sources, role)
             for role, name in s2_files.items()}
        require(sha256(p['zones']) == sha256(q['recovery_zones']), 'Patch and S2 recovery zones differ.')
        require(sha256(p['m02']) == sha256(q['recovery_m02']), 'Patch and S2 recovery observation support differs.')
        require(pd.read_csv(q['recovery_consistency'])['status'].eq('PASS').all(),
                'S2 recovery contains a failed numeric check.')
        method = json.loads(p['method'].read_text())
        source_rows = [r for r in method['inputs'] if r['role'] == 'optical']
        require(len(source_rows) == 1 and method['raster_metadata']['optical']['band_count'] == 14,
                'M01 provenance does not identify one original 14-band R03 export.')
        source = source_rows[0]
        records = pd.read_csv(q['source_records'], keep_default_na=False)
        change_records = records[(records['role'] == 'change')
                                 & records['file'].str.lower().str.endswith(('.tif', '.tiff'))]
        require(len(change_records) == 1, 'Ambiguous R03 provenance in S2 recovery.')
        recorded = change_records.iloc[0]
        require(str(recorded['sha256']).lower() == str(source['sha256']).lower(),
                'Patch/M01 and S2 recovery identify different R03 exports.')
        if cfg['change_file']:
            candidates = [Path(cfg['change_file'])]
        else:
            candidates = [Path(recorded['source_path']), Path(source['source_path']),
                          Path(cfg['export_dir']) / Path(recorded['file']).name]
        original_change = next((x for x in candidates if x.is_file()), None)
        require(original_change is not None,
                'Original R03 export was not found. Set change_file to its Drive path; do not use an RGB image.')
        require(sha256(original_change) == str(source['sha256']).lower(), 'Original R03 checksum mismatch.')
        q['change'] = work / 'change' / original_change.name
        q['change'].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_change, q['change'])
        require(sha256(q['change']) == str(source['sha256']).lower(), 'R03 local-copy checksum mismatch.')
        sources.append({'role': 'change', 'source_path': str(original_change),
                        'bytes': original_change.stat().st_size, 'sha256': sha256(q['change'])})
        # Copy only sidecars recorded by the recovery; do not introduce a new mask.
        for suffix in ['.msk', '.aux.xml', '.ovr']:
            sidecar = Path(str(original_change) + suffix)
            record_name = str(recorded['file']) + suffix
            matches = records[(records['role'] == 'change') & (records['file'] == record_name)]
            require(len(matches) <= 1, 'Duplicate R03 sidecar entries.')
            if len(matches):
                expected = matches.iloc[0]
                if not sidecar.is_file(): sidecar = Path(expected['source_path'])
                require(sidecar.is_file() and sha256(sidecar) == str(expected['sha256']).lower(),
                        f'Missing/changed recorded sidecar: {record_name}')
                destination = Path(str(q['change']) + suffix)
                shutil.copyfile(sidecar, destination)
                require(sha256(destination) == str(expected['sha256']).lower(), 'Sidecar copy mismatch.')
                sources.append({'role': 'change' + suffix, 'source_path': str(sidecar),
                                'bytes': sidecar.stat().st_size, 'sha256': sha256(destination)})
            else:
                require(not sidecar.is_file(), f'New unrecorded R03 sidecar: {sidecar}; use the immutable source.')

        # ------ Exact audited patch pixels and zone support ------
        m01, v01, ref = read_mask(p['m01'])
        m02, v02, ref2 = read_mask(p['m02'])
        require(ref == ref2 and np.array_equal(v01, v02 & (m02 == 1)), 'M01/M02 grid or mask mismatch.')
        require(m01.size <= cfg['max_grid_pixels'], 'Unexpectedly large audited raster.')
        with rasterio.open(p['labels']) as ds:
            require(ds.count == 1 and str(ds.crs) == ref['crs'] and ds.transform == ref['transform']
                    and ds.width == ref['width'] and ds.height == ref['height'], 'Patch-ID raster grid mismatch.')
            labels_ma = ds.read(1, masked=True)
            labels = np.asarray(labels_ma.data)
            require(np.array_equal(~np.ma.getmaskarray(labels_ma), v01), 'Patch-ID raster support mismatch.')
            require(np.issubdtype(labels.dtype, np.integer), 'Patch IDs must be integers.')
        require(np.array_equal((labels > 0) & v01, (m01 == 1) & v01), 'Label raster does not reproduce M01.')
        zones, geometries = read_zones(p['zones'], ref)
        require(not np.any((v01 | v02) & (zones == 0)), 'Audited observations outside the source zones.')
        primary = pd.read_csv(p['primary']).set_index('zone_id')
        for code, zid in [(5, 'R05'), (6, 'R06')]:
            n = int(np.count_nonzero((zones == code) & v01 & (m01 == 1)))
            require(n == integer(primary.loc[zid, 'concurrence_pixel_count'], 'M01 count'),
                    f'{zid}: M01 candidate-pixel count changed.')
        inventory = pd.read_csv(p['inventory'], keep_default_na=False)
        selected = inventory[inventory['patch_id'] == patch_id]
        require(len(selected) == 1, f'Expected exactly one record for {patch_id}.')
        record = selected.iloc[0]
        require(integer(record['connectivity'], 'Connectivity') == 4, 'Not a primary four-neighbour patch.')
        zid = str(record['zone_id']); code = {'R05': 5, 'R06': 6}[zid]
        target_full = v01 & (labels == integer(record['raster_id'], 'Raster ID'))
        ids = np.flatnonzero(target_full)
        require(len(ids) == integer(record['pixel_count'], 'Patch pixel count') and len(ids) > 0,
                'Selected patch pixel count mismatch.')
        pixel_m2 = abs(ref['transform'].a * ref['transform'].e)
        require(np.isclose(len(ids) * pixel_m2 / 1e6, float(record['area_km2']), atol=1e-10, rtol=0),
                'Selected patch area mismatch.')
        require(np.all((zones == code)[target_full]) and np.all((m01 == 1)[target_full]), 'Patch assigned incorrectly.')
        grid_key = json.dumps({'transform': list(ref['transform'])[:6], 'shape': list(m01.shape),
                               'crs': ref['crs']}, sort_keys=True).encode()
        digest = hashlib.sha256(grid_key + zid.encode() + np.asarray(ids, dtype='<u8').tobytes()).hexdigest()
        require(digest == record['pixel_membership_sha256'], 'Patch pixel-membership hash mismatch.')

        # ------ Bounded local crop, no grid resampling ------
        rr, cc = np.nonzero(target_full)
        pad = int(np.ceil(max(outers) / 20.0)) + 2
        r0, r1 = max(0, rr.min() - pad), min(ref['height'], rr.max() + pad + 1)
        c0, c1 = max(0, cc.min() - pad), min(ref['width'], cc.max() + pad + 1)
        window = Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
        crop_ref = {'crs': ref['crs'], 'transform': window_transform(window, ref['transform']),
                    'width': int(c1-c0), 'height': int(r1-r0)}
        sl = np.s_[r0:r1, c0:c1]
        target = target_full[sl]; z = zones[sl] == code
        joint = v01[sl]; local_m01 = m01[sl]
        pre, vp = read_named_window(q['pre'], REFLECTANCE + INDICES, crop_ref, 'PRE quantitative crop')
        post, vq = read_named_window(q['post'], REFLECTANCE + INDICES, crop_ref, 'POST quantitative crop')
        changes, vc = read_named_window(q['change'], ['d'+b for b in REFLECTANCE + INDICES] + ['spectral_rms'],
                                        crop_ref, 'R03 original quantitative export')
        require(np.all(vc['spectral_rms'][joint]), 'Joint support contains missing R03 RMS.')
        require(np.all(changes['spectral_rms'][target] >= 0.08), 'Selected M01 patch violates the RMS criterion.')
        require(np.all(changes['spectral_rms'][joint] >= 0), 'Valid RMS cannot be negative.')
        print(f'PATCH REPRODUCTION PASS: {patch_id}; {len(ids)} pixels; {len(ids)*pixel_m2/1e6:.4f} km2.')
        print('GRID PASS: original 20 m EPSG:32645 pixels; no resampling or rendered RGB values.')

        # ------ New, explicit local reference definition ------
        distance = ndi.distance_transform_edt(~target, sampling=(20.0, 20.0))
        regions = {'patch': target}
        reference_rows = []
        for outer in outers:
            ring = (distance > inner) & (distance <= outer) & ~target
            same_reach = ring & z
            observed = same_reach & joint
            reference = observed & (local_m01 == 0)
            name = f'local_nonM01_{inner:g}_{outer:g}m'
            regions[name] = reference
            ref_n = int(reference.sum())
            if ref_n < cfg['reference_small_n_warning']:
                warnings.append(f'{name}: only {ref_n} reference pixels; descriptive estimates may be unstable or undefined.')
            frame = np.zeros(target.shape, dtype=bool)
            if r0 == 0: frame[0, :] = True
            if r1 == ref['height']: frame[-1, :] = True
            if c0 == 0: frame[:, 0] = True
            if c1 == ref['width']: frame[:, -1] = True
            reference_rows.append({
                'region': name, 'inner_excluded_m': inner, 'outer_inclusive_m': outer,
                'geometric_ring_pixels_within_saved_grid': int(ring.sum()),
                'pixels_outside_target_reach': int((ring & ~z).sum()),
                'same_reach_ring_pixels': int(same_reach.sum()),
                'same_reach_unobserved_pixels': int((same_reach & ~joint).sum()),
                'other_M01_pixels_excluded': int((observed & (local_m01 == 1)).sum()),
                'reference_pixels': ref_n, 'reference_area_km2': ref_n * pixel_m2 / 1e6,
                'ring_reaches_saved_grid_edge': int(np.any(ring & frame)),
                'reference_status': 'NO_ELIGIBLE_PIXELS' if ref_n == 0 else
                                    'SMALL_DESCRIPTIVE_REFERENCE' if ref_n < cfg['reference_small_n_warning'] else
                                    'DESCRIPTIVE_REFERENCE_NOT_UNAFFECTED_CONTROL',
            })
            require(not np.any(reference & ((local_m01 != 0) | ~joint | ~z)), 'Reference exclusions failed.')
        main_name = f'local_nonM01_{inner:g}_{primary_outer:g}m'
        all_domain = target | regions[f'local_nonM01_{inner:g}_{max(outers):g}m']

        # ------ Recheck POST-minus-PRE direction and RMS ------
        consistency_rows = []
        def compare(metric, calculated, available):
            need = all_domain & vc[metric]
            take = need & available
            errors = np.abs(calculated[take] - changes[metric][take])
            tolerance = cfg['absolute_tolerance'] + cfg['relative_tolerance'] * np.abs(changes[metric][take])
            n_missing = int((need & ~available).sum()); n_bad = int((errors > tolerance).sum())
            consistency_rows.append({'metric': metric, 'compared_pixels': int(take.sum()),
                'R03_valid_without_source_values': n_missing, 'outside_numeric_tolerance': n_bad,
                'max_absolute_error': float(errors.max()) if len(errors) else np.nan,
                'status': 'PASS' if take.any() and n_missing == 0 and n_bad == 0 else 'FAIL'})
        paired_valid, raw_deltas = {}, {}
        squared = np.zeros(target.shape, dtype=np.float64)
        counts = np.zeros(target.shape, dtype=np.uint8)
        for name in REFLECTANCE:
            available = vp[name] & vq[name]
            raw_deltas[name] = post[name] - pre[name]
            calculated = np.maximum(post[name], 0) - np.maximum(pre[name], 0)
            compare('d'+name, calculated, available)
            squared += np.where(available, calculated**2, 0)
            counts += available.astype(np.uint8)
            paired_valid[name] = joint & available & vc['d'+name]
        for name in INDICES:
            available = vp[name] & vq[name]
            compare('d'+name, post[name] - pre[name], available)
            paired_valid[name] = joint & available & vc['d'+name]
        calculated_rms = np.sqrt(np.divide(squared, counts, out=np.zeros_like(squared), where=counts > 0))
        compare('spectral_rms', calculated_rms, counts > 0)
        consistency = pd.DataFrame(consistency_rows)
        consistency.to_csv(out / 'numeric_consistency.csv', index=False)
        require(consistency['status'].eq('PASS').all(),
                'PRE/POST versus R03 consistency failed; see numeric_consistency.csv. No interpretation assigned.')
        print('NUMERIC CONSISTENCY PASS: correct POST-minus-PRE signs and original spectral RMS.')

        # ------ Explicit paired denominators, full distributions ------
        stats_rows, class_rows, threshold_rows, paired_rows, reflectance_rows = [], [], [], [], []
        metric_values = {k: changes[k] for k in ['dNDVI', 'dMNDWI', 'dNBR', 'spectral_rms']}
        metric_valid = {f'd{b}': paired_valid[b] for b in MAIN_INDICES}
        metric_valid['spectral_rms'] = joint & vc['spectral_rms']
        for region_name, region in regions.items():
            region_n = int(region.sum())
            for name, values in metric_values.items():
                take = region & metric_valid[name]
                summary = summarize(values[take])
                stats_rows.append({'patch_id': patch_id, 'region': region_name, 'metric': name,
                    'region_pixels': region_n, 'valid_pct_of_region': 100*summary['n']/region_n if region_n else np.nan,
                    'missing_metric_pixels': region_n-summary['n'], **summary})
                if name in CLASS_DEFINITIONS:
                    for entry in proportions(values[take], name):
                        class_rows.append({'patch_id': patch_id, 'region': region_name,
                            'metric_valid_pixels': int(take.sum()), 'region_pixels': region_n,
                            'class_area_km2': entry['pixels'] * pixel_m2 / 1e6, **entry})
            for metric, label, op, cut in THRESHOLD_TESTS:
                take = region & metric_valid[metric]
                selected_n = int(np.count_nonzero(take & threshold_mask(metric_values[metric], op, cut)))
                n = int(take.sum())
                threshold_rows.append({'patch_id': patch_id, 'region': region_name, 'metric': metric,
                    'criterion': label, 'operator': op, 'threshold': cut, 'metric_valid_pixels': n,
                    'criterion_pixels': selected_n, 'criterion_pct': 100*selected_n/n if n else np.nan})
            for name in INDICES:
                take = region & paired_valid[name]
                a, b, d = summarize(pre[name][take]), summarize(post[name][take]), summarize(changes['d'+name][take])
                paired_rows.append({'patch_id': patch_id, 'region': region_name, 'index': name,
                    'paired_pixels': int(take.sum()), 'missing_paired_pixels': region_n-int(take.sum()),
                    **{f'pre_{k}': a[k] for k in ['mean','median','q25','q75']},
                    **{f'post_{k}': b[k] for k in ['mean','median','q25','q75']},
                    **{f'delta_{k}': d[k] for k in ['mean','median','q25','q75','iqr']}})
            for name in REFLECTANCE:
                take = region & paired_valid[name]
                a, b = summarize(pre[name][take]), summarize(post[name][take])
                raw, stored = summarize(raw_deltas[name][take]), summarize(changes['d'+name][take])
                reflectance_rows.append({'patch_id': patch_id, 'region': region_name, 'band': name,
                    'paired_pixels': int(take.sum()),
                    **{f'pre_{k}': a[k] for k in ['mean','median','q25','q75']},
                    **{f'post_{k}': b[k] for k in ['mean','median','q25','q75']},
                    **{f'raw_post_minus_pre_{k}': raw[k] for k in ['mean','median','q25','q75']},
                    **{f'stored_R03_clamped_delta_{k}': stored[k] for k in ['mean','median','q25','q75']},
                    'pre_negative_value_pixels': int((take & (pre[name] < 0)).sum()),
                    'post_negative_value_pixels': int((take & (post[name] < 0)).sum())})
        stats_df, class_df, thresholds_df = map(pd.DataFrame, [stats_rows, class_rows, threshold_rows])
        comparison_rows = []
        for region_name in regions:
            if region_name == 'patch': continue
            for metric, criterion, _, _ in THRESHOLD_TESTS:
                a = thresholds_df[(thresholds_df.region == 'patch') & (thresholds_df.criterion == criterion)].iloc[0]
                b = thresholds_df[(thresholds_df.region == region_name) & (thresholds_df.criterion == criterion)].iloc[0]
                sa = stats_df[(stats_df.region == 'patch') & (stats_df.metric == metric)].iloc[0]
                sb = stats_df[(stats_df.region == region_name) & (stats_df.metric == metric)].iloc[0]
                comparison_rows.append({'patch_id': patch_id, 'reference_region': region_name,
                    'metric': metric, 'criterion': criterion, 'patch_n': int(a.metric_valid_pixels),
                    'reference_n': int(b.metric_valid_pixels), 'patch_median': sa['median'],
                    'reference_median': sb['median'], 'median_difference_patch_minus_reference': sa['median']-sb['median'],
                    'patch_criterion_pct': a.criterion_pct, 'reference_criterion_pct': b.criterion_pct,
                    'difference_percentage_points': a.criterion_pct-b.criterion_pct,
                    'status': 'DESCRIPTIVE_SELECTION_CONDITIONED' if b.metric_valid_pixels else 'NO_REFERENCE'})
        compare_df = pd.DataFrame(comparison_rows)

        # ------ Per-pixel data, no hidden masks ------
        if cfg['write_pixel_table']:
            domain = all_domain
            local_r, local_c = np.nonzero(domain)
            world_x, world_y = crop_ref['transform'] * (local_c+0.5, local_r+0.5)
            lon, lat = Transformer.from_crs(32645, 4326, always_xy=True).transform(world_x, world_y)
            data = {'patch_id': patch_id, 'sample_role': np.where(target[domain], 'target_patch', 'nearby_nonM01_reference'),
                    'grid_row': local_r+r0, 'grid_col': local_c+c0, 'easting_m': world_x,
                    'northing_m': world_y, 'longitude': lon, 'latitude': lat,
                    'nearest_target_pixel_centre_m': distance[domain]}
            for name, region in regions.items(): data['in_'+name] = region[domain].astype(np.uint8)
            for role, values, valid in [('PRE',pre,vp), ('POST',post,vq), ('R03',changes,vc)]:
                for band in values:
                    data[f'{role}_{band}'] = np.where(valid[band][domain], values[band][domain], np.nan)
            pd.DataFrame(data).to_csv(out / 'patch_and_reference_pixel_values.csv', index=False)

        # ------ QGIS support raster: target + disjoint distance bins ------
        membership = np.full(target.shape, 255, dtype=np.uint8)
        membership[z & joint] = 0
        membership[target] = 1
        previous = inner
        selection_legend = [
            {'value': 0, 'meaning': 'Same reach and jointly observed; not a selected sample'},
            {'value': 1, 'meaning': 'Exact target M01 patch'},
            {'value': 255, 'meaning': 'Outside target reach or not jointly observed'},
        ]
        for i, outer in enumerate(outers, start=2):
            eligible = regions[f'local_nonM01_{inner:g}_{outer:g}m'] & (distance > previous)
            membership[eligible] = i
            selection_legend.append({'value': i, 'meaning': f'Eligible non-M01 pixels: {previous:g} < distance <= {outer:g} m'})
            previous = outer
        profile = dict(driver='GTiff', width=crop_ref['width'], height=crop_ref['height'], count=1,
                       dtype='uint8', crs=crop_ref['crs'], transform=crop_ref['transform'], nodata=255,
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256)
        raster_path = out / 'patch_and_local_reference_selection.tif'
        with rasterio.open(raster_path, 'w', **profile) as ds:
            ds.write(membership, 1); ds.set_band_description(1, 'sample_membership_NOT_change_or_confidence')
        with rasterio.open(raster_path) as ds:
            require(np.array_equal(ds.read(1), membership), 'Selection-raster readback failure.')
        pd.DataFrame(selection_legend).to_csv(out / 'reference_selection_legend.csv', index=False)
        # Numeric local map retaining the original R03 values on joint support.
        names = list(metric_values)
        profile.update(dtype='float32', count=len(names), nodata=-9999.0, predictor=3)
        with rasterio.open(out / 'directional_metrics_local_context.tif', 'w', **profile) as ds:
            for i, name in enumerate(names, 1):
                good = z & metric_valid[name]
                ds.write(np.where(good, metric_values[name], -9999).astype('float32'), i)
                ds.set_band_description(i, name)
        pd.DataFrame([{'band': i+1, 'name': name} for i,name in enumerate(names)]).to_csv(
            out / 'directional_metrics_band_order.csv', index=False)

        # Selected-patch outline: actual cell footprints, no smooth/convex-hull replacement.
        parts = [shape(g) for g, _ in shapes(target.astype('uint8'), mask=target,
                                            transform=crop_ref['transform'], connectivity=4)]
        if all(g.is_valid for g in parts):
            patch_geometry = unary_union(parts)
        else:
            # Exact union of the same grid cells; never buffer(0) or smooth.
            cells = []
            for r,c in zip(*np.nonzero(target)):
                x,y = crop_ref['transform'] * (int(c),int(r))
                cells.append(box(x, y-20, x+20, y))
            patch_geometry = unary_union(cells)
        require(patch_geometry.is_valid and np.isclose(patch_geometry.area, len(ids)*pixel_m2, atol=1e-5, rtol=0),
                'Target polygon area mismatch.')
        raster_back = rasterize([(mapping(patch_geometry),1)], out_shape=target.shape,
                                transform=crop_ref['transform'], all_touched=False, dtype='uint8') > 0
        require(np.array_equal(raster_back, target), 'Target polygon changed the original pixel footprint.')
        geographic = transform_geometry(Transformer.from_crs(32645,4326,always_xy=True).transform, patch_geometry)
        write_json(out / 'selected_patch_outline.geojson', {'type':'FeatureCollection', 'features':[
            {'type':'Feature', 'geometry':mapping(geographic), 'properties':{
                'patch_id': patch_id, 'pixel_count':len(ids), 'area_km2':len(ids)*pixel_m2/1e6,
                'status':'numerical_characterization_only_no_validation_assignment'}}]})

        # ------ Write tables/provenance, never edit validation_status ------
        identity = selected.copy()
        identity['membership_check'] = 'PASS'
        identity['directional_analysis_status'] = 'characterization_only_review_status_not_changed'
        tables = {
            'patch_identity.csv': identity,
            'directional_metric_summary.csv': stats_df,
            'directional_class_proportions.csv': class_df,
            'directional_threshold_proportions.csv': thresholds_df,
            'paired_PRE_POST_indices.csv': pd.DataFrame(paired_rows),
            'paired_reflectance_summary.csv': pd.DataFrame(reflectance_rows),
            'patch_vs_local_nonM01_reference.csv': compare_df,
            'local_reference_coverage.csv': pd.DataFrame(reference_rows),
            'input_checksums.csv': pd.DataFrame(sources),
        }
        for filename, table in tables.items(): table.to_csv(out / filename, index=False)
        limitations = [
            'This is spectral characterization of a detection-selected patch, not independent validation.',
            'R05_P0001 was targeted after inspection and by patch size, not randomly sampled.',
            'M01 conditions on SAR Class3 and RMS >=0.08. Patch/reference RMS contrasts are partly selected by design.',
            'Non-M01 reference pixels are not verified unaffected controls; land cover, terrain and acquisition timing are not matched.',
            'Reference windows are nested, exploratory descriptive settings, not independent replicates.',
            'Distances are to target pixel centres, not surveyed banks or continuous polygon edges.',
            'Internal patch holes can enter the nearby reference when they meet the distance and observation criteria.',
            'PRE/POST and indices are the same composite sources used for detection; they are not separate validations.',
            'Optical gaps are missing, never zeros or stable pixels. Per-metric denominators are explicit.',
            'Negative dNDVI is index decline, not automatic vegetation destruction. dMNDWI/dNBR do not uniquely identify a process.',
            'No p-values, independent-pixel confidence intervals, event causality, damage labels or posterior probabilities.',
            'Valid masks and numeric consistency do not guarantee freedom from cloud, haze, shadow or compositing artefacts.',
            'Delta medians are medians of per-pixel differences, not differences of separate PRE/POST medians.',
        ]
        for flag in ['any_gap_or_boundary_flag','touches_joint_gap_8n','touches_outer_buffer_edge_8n']:
            if flag in record and integer(record[flag], flag):
                warnings.append(f'Target patch has {flag}=1; its footprint may be observation/boundary limited.')
        code_file = out / 'directional_code_used.py'
        if source_code_path and Path(source_code_path).is_file(): shutil.copyfile(source_code_path, code_file)
        else:
            try:
                from IPython import get_ipython
                history = get_ipython().history_manager.input_hist_raw
                text = next((s for s in reversed(history) if 'def run_patch_directional(' in s), None)
                if text: code_file.write_text(text, encoding='utf-8')
            except Exception: pass
        metadata = {'version':VERSION, 'created_utc':datetime.now(timezone.utc).isoformat(), 'settings':cfg,
            'patch_pixel_membership_sha256':digest, 'source_files':sources,
            'reference_method':'Euclidean centre distances; inner exclusive/outer inclusive; same reach, M01 valid=0 only',
            'grid':{'crs':ref['crs'],'transform':list(ref['transform'])[:6], 'shape':list(m01.shape)},
            'local_window':{'row_offset':int(r0),'col_offset':int(c0),'rows':int(r1-r0),'cols':int(c1-c0)},
            'quantiles':'NumPy method=linear; IQR=Q75-Q25', 'classification_upper_limits_inclusive':True,
            'numerical_tolerances':{'absolute':cfg['absolute_tolerance'],'relative':cfg['relative_tolerance']},
            'same_indices_pre_post_delta_use_same_paired_pixels':True,
            'primary_reference_region':main_name,'limitations':limitations,'warnings':warnings,
            'old_files_or_review_labels_modified':False,
            'code_sha256':sha256(code_file) if code_file.is_file() else None,
            'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                'rasterio':rasterio.__version__,'GDAL':rasterio.__gdal_version__,'scipy':scipy.__version__,
                'shapely':shapely.__version__,'pyproj':pyproj.__version__}}
        write_json(out / 'directional_method_and_inputs.json', metadata)
        readme = f'''AUDITED PATCH CHARACTERIZATION {VERSION}\nTarget: {patch_id}\n\nPRIMARY REFERENCE: {main_name}\nDistances are to the nearest target pixel centre, in EPSG:32645 metres.\nReference regions contain only SAME-REACH jointly observed non-M01 pixels.\nAll other M01 patches and missing observations are excluded. Internal holes may\nbe included when they meet the distance rule. The rings are nested alternatives.\n\nTables:\n- directional_metric_summary: medians, Q25/Q75, IQR, mean, SD, range and counts.\n- directional_class_proportions: five mutually exclusive signed display intervals.\n- directional_threshold_proportions: exact strict/non-strict criteria recorded.\n- paired_PRE_POST_indices: identical paired denominators for PRE, POST and delta.\n- paired_reflectance_summary: raw POST-PRE separate from stored zero-clamped deltas.\n- patch_vs_local_nonM01_reference: descriptive contrasts only.\n- local_reference_coverage: missing/excluded pixels, sample sizes and grid-edge checks.\n\nQGIS:\nUse selected_patch_outline.geojson with NO FILL and clear the yellow selection.\npatch_and_local_reference_selection.tif shows the samples used, NOT an impact map.\ndirectional_metrics_local_context.tif holds dNDVI,dMNDWI,dNBR,spectral_rms, in that order.\n\nIMPORTANT:\n''' + '\n'.join('- '+text for text in limitations) + '\n'
        (out / 'README_directional_review.txt').write_text(readme, encoding='utf-8')
        output_files = sorted(path for path in out.iterdir() if path.is_file())
        pd.DataFrame([{'file':path.name,'bytes':path.stat().st_size,'sha256':sha256(path)}
                      for path in output_files]).to_csv(out / 'output_checksums.csv', index=False)
        bundle = out / f'{patch_id}_directional_characterization_bundle.zip'
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out.iterdir()):
                if path.is_file() and path != bundle: zf.write(path, path.name)
        with zipfile.ZipFile(bundle) as zf: require(zf.testzip() is None, 'ZIP integrity failure.')
        drive_destination = None
        if cfg['save_to_drive']:
            drive_destination = Path(cfg['drive_output_root']) / out.name
            drive_destination.mkdir(parents=True, exist_ok=False)
            for path in sorted(out.iterdir()):
                if path.is_file():
                    dest = drive_destination / path.name
                    shutil.copyfile(path, dest)
                    require(sha256(path) == sha256(dest), f'Drive copy checksum mismatch: {path.name}')
        print('\nPATCH DIRECTIONAL SUMMARY — STORED POST MINUS PRE')
        show(stats_df[stats_df.region == 'patch'][['metric','n','missing_metric_pixels','median','q25','q75','iqr','minimum','maximum']].round(5))
        print('\nPATCH DIRECTIONAL CLASS PROPORTIONS')
        show(class_df[class_df.region == 'patch'][['metric','class_label','interval','pixels','pct_of_metric_valid_pixels']].round(4))
        print('\nPATCH VERSUS LOCAL NON-M01 REFERENCE — PRIMARY WINDOW')
        show(compare_df[compare_df.reference_region == main_name].round(5))
        print('\nLOCAL REFERENCE AVAILABILITY')
        show(pd.DataFrame(reference_rows))
        for warning in warnings: print('REVIEW:', warning)
        print('\nSaved locally:', out)
        if drive_destination: print('Saved and checksum-verified on Drive:', drive_destination)
        print('No Earth Engine tasks. No original pixels, geometry or validation_status were changed.')
        return {'local_dir':out,'drive_dir':drive_destination,'summary':stats_df,'classes':class_df,
                'comparison':compare_df,'coverage':pd.DataFrame(reference_rows),'consistency':consistency}
    except Exception as exc:
        write_json(out / 'DIRECTIONAL_FAILED.json', {'error_type':type(exc).__name__,'error':str(exc),
                    'status':'NOT_ACCEPTED_FOR_INTERPRETATION','old_files_modified':False})
        print(f'Analysis stopped: {exc}\nLocal diagnostics: {out}')
        raise


if __name__ == '__main__':
    PATCH_DIRECTIONAL_RESULTS = run_patch_directional(
        source_code_path=globals().get('PATCH_DIRECTIONAL_SCRIPT_PATH') or globals().get('__file__'))
