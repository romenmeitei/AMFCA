# ============================================================
# LHENDE R05_P0001: DATED SENTINEL-2 QUALITY / PERSISTENCE AUDIT
# v1.0.1 -- Metadata serialization fix; scientific settings unchanged
# NEW scene-level diagnostic, NOT a replacement of M01.
# Historical settings checked against the user's v1_4_2 original source:
# event=2026-08-26T02:52:10Z, original post end=2026-09-02, CP<55, cosine>0.15.
# The stricter SCL/CS+ gates, 60m adjacency exclusion and 70% coverage are NEW audit settings.
# No new damage labels; no Earth Engine batch export tasks.
# ============================================================
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from contextlib import ExitStack
import hashlib
import html
import importlib.util
import io
import json
import math
import re
import shutil
import subprocess
import sys
import time
import zipfile

SCENE_AUDIT_CONFIG = {
    'ee_project': 'woven-name-441217-g5',
    'patch_id': 'R05_P0001',
    'patch_dir': '/content/drive/MyDrive/Lhende_2026_Candidate_Patches/M01_patch_inventory_20260911T082650777727Z',
    's2_recovery_dir': '/content/drive/MyDrive/Lhende_2026_S2_Review/S2_export_recovery_20260911T093217241209Z',
    'original_audit_dir': '/content/drive/MyDrive/Lhende_2026_GEE_Audit',
    'event_time_utc': '2026-08-26T02:52:10Z',  # supplied study anchor, not independently verified
    'original_primary_pre_start': '2026-08-22T00:00:00Z',
    'original_fallback_pre_start': '2026-08-10T00:00:00Z',
    'original_fallback_pre_end': '2026-08-16T00:00:00Z',
    'original_post_end_exclusive': '2026-09-02T00:00:00Z',
    # Expanded, explicitly NEW review window. End is frozen at execution time.
    'review_start_utc': '2026-08-01T00:00:00Z',
    'review_end_exclusive_utc': '',  # blank = UTC time when the run begins
    'context_margin_m': 800,
    'cloud_probability_max': 55.0,  # retained original threshold, pass < 55
    'min_cosine_illumination': 0.15,  # retained original illumination screen
    # NEW audit quality settings: fixed before examining dated results.
    'cloud_score_cdf_min': 0.60,
    'strict_scl_classes': [4, 5, 6],  # vegetation, nonvegetated, water; not vegetation-only
    'cloud_shadow_buffer_m': 60.0,
    'maximum_solar_zenith_deg': 70.0,
    'minimum_patch_coverage_fraction': 0.70,
    'primary_quality_mode': 'strict_dual_QA',
    'directional_NDVI_decline_threshold': -0.15,
    'composite_match_abs_tolerance': 0.000002,
    'dem_collection': 'COPERNICUS/DEM/GLO30_2024_1',
    'maximum_scene_assets': 160,  # safety limit: stop, never silently truncate
    'maximum_chip_pixels': 250000,
    'download_attempts': 3,
    'http_timeout_seconds': 180,
    'save_to_drive': True,
    'local_output_root': '/content/Lhende_2026_Dated_Scene_Audit',
    'drive_output_root': '/content/drive/MyDrive/Lhende_2026_Dated_Scene_Audit',
    # To recover an interrupted run, paste its own output folder here.
    # This reuses its frozen inventory/time cutoff; it does not add new dates.
    'resume_drive_run_dir': '',
    'write_scene_RGB': True,
    'display_min': 0.02,
    'display_max': 0.35,
    'display_gamma': 1.1,
}
VERSION = '1.0.1'
SR_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
CP_COLLECTION = 'COPERNICUS/S2_CLOUD_PROBABILITY'
CS_COLLECTION = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
CRS = 'EPSG:32645'
REFLECTANCE = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
CHIP_BANDS = REFLECTANCE + ['SCL', 'edge_valid', 'cloud_probability', 'cs', 'cs_cdf', 'AOT', 'WVP']
MODES = ['scl_illumination', 'cloud_probability_QA', 'cloud_score_QA', 'strict_dual_QA', 'legacy_like_diagnostic']
NODATA = -9999.0
# Official primary sources, also saved in the run manifest.
DOCS = {
    'sr': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED',
    'cloud_probability': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY',
    'cloud_score_plus': 'https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED',
    'scl_limitations': 'https://sentiwiki.copernicus.eu/web/s2-products',
    'scl_processing': 'https://sentiwiki.copernicus.eu/web/s2-processing',
    'small_downloads': 'https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl',
    'dem': 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30_2024_1',
    'metadata_fix_reference': 'https://developers.google.com/earth-engine/docs/release-notes#july-31-2013',
    'image_get': 'https://developers.google.com/earth-engine/apidocs/ee-image-get',
    'image_id': 'https://developers.google.com/earth-engine/apidocs/ee-image-id',
}


def dependencies():
    modules = {'numpy':'numpy', 'pandas':'pandas', 'rasterio':'rasterio', 'shapely':'shapely',
               'pyproj':'pyproj', 'scipy':'scipy', 'requests':'requests', 'PIL':'Pillow',
               'matplotlib':'matplotlib', 'ee':'earthengine-api'}
    missing = [package for module, package in modules.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def utc(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)
    text = str(value).strip().replace('Z', '+00:00')
    return datetime.fromisoformat(text).replace(tzinfo=datetime.fromisoformat(text).tzinfo or timezone.utc).astimezone(timezone.utc)


def iso(value):
    return utc(value).isoformat().replace('+00:00', 'Z')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_safe(x):
    import numpy as np
    if isinstance(x, dict): return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return json_safe(x.tolist())
    if isinstance(x, np.generic): return json_safe(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    if isinstance(x, datetime): return iso(x)
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def safe_error(exc):
    # Do not log signed download URLs or credentials.
    return re.sub(r'https?://\S+', '[URL redacted]', f'{type(exc).__name__}: {exc}')[:1600]


def resolved_times(cfg, now=None):
    now = utc(now or datetime.now(timezone.utc))
    start = utc(cfg['review_start_utc'])
    requested = utc(cfg['review_end_exclusive_utc']) if cfg['review_end_exclusive_utc'] else now
    end = min(requested, now)
    event = utc(cfg['event_time_utc'])
    require(start < event < end, 'The resolved review window must include observations before and after the supplied event time.')
    return {'start_utc': iso(start), 'end_exclusive_utc': iso(end), 'event_time_utc': iso(event),
            'run_as_of_utc': iso(now), 'requested_end_was_future': requested > now}


def checked_files(directory, names, destination, log_rows, prefix):
    """Verify upstream manifests before copying source files; never modify originals."""
    import pandas as pd
    directory, destination = Path(directory), Path(destination)
    manifest_path = directory / 'output_checksums.csv'
    require(manifest_path.is_file(), f'Missing upstream checksum manifest: {manifest_path}')
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    require({'file', 'sha256', 'bytes'} <= set(frame), f'Unexpected manifest format: {manifest_path}')
    require(not frame['file'].duplicated().any(), 'Duplicate paths in upstream checksum manifest.')
    frame = frame.set_index('file')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, destination / 'upstream_output_checksums.csv')
    result = {}
    for name in names:
        require(name in frame.index, f'{name} is absent from upstream checksums.')
        path = directory / name
        require(path.is_file(), f'Missing input: {path}')
        require(path.stat().st_size == int(frame.loc[name, 'bytes']), f'Size changed: {path}')
        digest = sha256(path)
        require(digest == str(frame.loc[name, 'sha256']).lower(), f'Checksum changed: {path}')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha256(target) == digest, 'Local copy checksum failed.')
        result[name] = target
        log_rows.append({'role': prefix + '/' + name, 'source_path': str(path),
                         'sha256': digest, 'bytes': path.stat().st_size})
    return result


def load_patch_inputs(cfg, out):
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.windows import Window, transform as wt
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union, transform
    from pyproj import Transformer
    logs = []
    names = ['patch_ids_4n.tif', 'P01_candidate_patch_inventory_4n.csv',
             'M01_concurrence_RMS008_audited.tif', 'M02_joint_optical_support.tif',
             'river_reference_buffers_500m_used.geojson']
    src = checked_files(cfg['patch_dir'], names, out/'source_inputs'/'patch', logs, 'patch')
    table = pd.read_csv(src['P01_candidate_patch_inventory_4n.csv'])
    row = table.loc[table['patch_id'].eq(cfg['patch_id'])]
    require(len(row) == 1, 'Patch ID must match exactly one original inventory row.')
    record = row.iloc[0].to_dict()
    require(int(record['connectivity']) == 4, 'Use the primary four-neighbour patch inventory.')
    with rasterio.open(src['patch_ids_4n.tif']) as ds:
        require(str(ds.crs) == CRS and ds.count == 1, 'Unexpected patch ID raster CRS/bands.')
        t = ds.transform
        require(np.allclose([t.a,t.b,t.d,t.e], [20,0,0,-20], atol=1e-9, rtol=0), 'Expected north-up 20 m audit grid.')
        require(ds.width*ds.height<=25_000_000, 'Unexpectedly large original patch-ID raster.')
        labels = ds.read(1)
        target_full = labels == int(record['raster_id'])
        ids = np.flatnonzero(target_full)
        require(len(ids) == int(record['pixel_count']), 'Patch pixel-count mismatch.')
        grid_key = json.dumps({'transform': list(t)[:6], 'shape': list(labels.shape), 'crs': CRS}, sort_keys=True).encode()
        digest = hashlib.sha256(grid_key + str(record['zone_id']).encode() + np.asarray(ids, dtype='<u8').tobytes()).hexdigest()
        require(digest == record['pixel_membership_sha256'], 'Patch membership hash mismatch.')
        rr, cc = np.where(target_full)
        pad = int(math.ceil(cfg['context_margin_m']/20))
        # A review context window, NOT new detection geometry. Edge limitations recorded.
        r0, r1 = max(0,rr.min()-pad), min(ds.height,rr.max()+pad+1)
        c0, c1 = max(0,cc.min()-pad), min(ds.width,cc.max()+pad+1)
        window = Window(int(c0), int(r0), int(c1-c0), int(r1-r0))
        ref = {'transform': wt(window,t), 'width': int(window.width), 'height': int(window.height), 'crs': CRS}
        full_ref = {'transform': t, 'width': ds.width, 'height': ds.height, 'crs': CRS}
    target = target_full[r0:r1,c0:c1]
    require(target.size <= cfg['maximum_chip_pixels'], 'Review chip exceeds configured download safety limit.')
    require(target.any(), 'Empty patch.')
    for name, expected in [('M01_concurrence_RMS008_audited.tif',1), ('M02_joint_optical_support.tif',1)]:
        with rasterio.open(src[name]) as ds:
            require(ds.transform == full_ref['transform'] and str(ds.crs) == CRS and
                    (ds.width,ds.height)==(full_ref['width'],full_ref['height']), 'Upstream mask grid mismatch.')
            ar = ds.read(1, window=window, masked=True)
            require(np.all(~np.ma.getmaskarray(ar)[target]) and np.all(ar.data[target] == expected), 'Patch not fully supported in M01/M02.')
    parts = [shape(g) for g,v in shapes(target.astype('uint8'), mask=target, connectivity=4, transform=ref['transform']) if v==1]
    geom = unary_union(parts)
    require(geom.is_valid and abs(geom.area-len(ids)*400)<1e-5, 'Patch polygon/pixel mismatch.')
    geographic = transform(Transformer.from_crs(32645,4326,always_xy=True).transform, geom)
    write_json(out/'patch_outline.geojson', {'type':'FeatureCollection', 'features':[{'type':'Feature',
        'geometry':mapping(geographic), 'properties':{'patch_id':cfg['patch_id'], 'pixel_count':len(ids),
        'area_km2':len(ids)*.0004, 'review_status':'not_changed_by_dated_scene_audit'}}]})
    # Cropped quantitative PRE/POST retained pixels, only for consistency attribution screens.
    s2names = ['V04_PRE_original_export_bands_R05_R06.tif', 'V05_POST_original_export_bands_R05_R06.tif', 'S2_recovery_manifest.json']
    s2 = checked_files(cfg['s2_recovery_dir'], s2names, out/'source_inputs'/'composites', logs, 'recovered_composites')
    composites = {}
    for phase, name in [('PRE',s2names[0]),('POST',s2names[1])]:
        with rasterio.open(s2[name]) as ds:
            rel = (~ds.transform)*ref['transform']
            require(str(ds.crs)==CRS and np.allclose([rel.a,rel.b,rel.d,rel.e],[1,0,0,1],atol=1e-7,rtol=0) and
                    np.allclose([rel.c,rel.f],np.round([rel.c,rel.f]),atol=1e-6,rtol=0), 'Composite grid differs from patch.')
            descriptions = list(ds.descriptions)
            require(all(descriptions.count(b)==1 for b in REFLECTANCE), 'Cannot identify all six saved reflectance bands.')
            inds = [descriptions.index(b)+1 for b in REFLECTANCE]
            ar = ds.read(inds, window=Window(int(round(rel.c)),int(round(rel.f)),ref['width'],ref['height']),
                         boundless=True, masked=True).astype('float64')
            require(all(ds.scales[b-1]==1 and ds.offsets[b-1]==0 for b in inds), 'Unexpected saved reflectance scale/offset.')
            a = ar.filled(np.nan)
            a[a==NODATA] = np.nan
            require(np.all(np.isfinite(a[:,target])), 'Target lacks the recovered paired quantitative reflectance.')
            composites[phase] = a
    pd.DataFrame(logs).to_csv(out/'input_checksums.csv',index=False)
    write_json(out/'patch_identity.json', record)
    print(f"PATCH PASS: {cfg['patch_id']}; {len(ids)} pixels; {len(ids)*.0004:.4f} km2. Original membership unchanged.")
    return ref, target, geographic, composites, record


def read_original_records(cfg, out):
    """Read only the known audit directory / named ZIP. No broad Drive search."""
    import pandas as pd
    root = Path(cfg['original_audit_dir'])
    zip_path = root/'Lhende_2026_local_audit_bundle.zip'
    inventory = {'PRE':None, 'POST':None}
    records = []
    dest = out/'source_inputs'/'original_inventory'
    dest.mkdir(parents=True,exist_ok=True)
    def read_bytes(name):
        direct = root/name
        if direct.is_file(): return direct.read_bytes(), str(direct)
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path) as z:
                matches = [n for n in z.namelist() if Path(n).name==name and not n.endswith('/')]
                require(len(matches)<=1, f'Ambiguous {name} inside saved audit ZIP.')
                if matches:
                    require(z.getinfo(matches[0]).file_size<20_000_000, 'Unexpectedly large audit metadata file.')
                    return z.read(matches[0]), str(zip_path)+'::'+matches[0]
        return None, None
    for phase, filename in [('PRE','s2_pre_scene_inventory.csv'),('POST','s2_post_scene_inventory.csv')]:
        raw, source = read_bytes(filename)
        if raw is not None:
            (dest/filename).write_bytes(raw)
            try:
                frame = pd.read_csv(io.BytesIO(raw))
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            ids = None
            if 'system_index' in frame:
                ids = {str(x).split('/')[-1] for x in frame['system_index'].dropna() if str(x).strip()}
            inventory[phase] = ids
            records.append({'role':phase, 'source':source, 'sha256':hashlib.sha256(raw).hexdigest(),
                            'rows':len(frame), 'system_index_available':ids is not None})
        else:
            records.append({'role':phase,'source':None,'system_index_available':False})
    raw, source = read_bytes('run_manifest.json')
    if raw is not None:
        (dest/'run_manifest.json').write_bytes(raw)
        manifest = json.loads(raw)
        if manifest.get('event_time_utc'):
            require(utc(manifest['event_time_utc'])==utc(cfg['event_time_utc']), 'Saved event time differs; resolve configuration before auditing.')
        windows = manifest.get('date_windows',{})
        mapping_keys = {'s2_primary_pre_start':'original_primary_pre_start','s2_fallback_pre_start':'original_fallback_pre_start',
                        's2_fallback_pre_end':'original_fallback_pre_end','s2_post_end':'original_post_end_exclusive'}
        for old,new in mapping_keys.items():
            if windows.get(old): cfg[new] = iso(utc(windows[old]))
        records.append({'role':'manifest','source':source,'sha256':hashlib.sha256(raw).hexdigest()})
    write_json(out/'original_inventory_status.json', records)
    return inventory


def classify_time(timestamp, cfg):
    when = utc(timestamp)
    if when < utc(cfg['event_time_utc']): return 'PRE_EVENT'
    if when < utc(cfg['original_post_end_exclusive']): return 'ORIGINAL_POST_WINDOW'
    return 'LATER_FOLLOWUP'


def initialize_ee(cfg):
    import ee
    try:
        ee.Initialize(project=cfg['ee_project'])
        ee.Number(1).getInfo()
    except Exception:
        print('Earth Engine needs authentication. Use the account authorized for the project.')
        ee.Authenticate()
        ee.Initialize(project=cfg['ee_project'])
        ee.Number(1).getInfo()
    print('Earth Engine initialized:', cfg['ee_project'])


def normalise_scene_inventory_record(record, source_collection=None, position=None):
    """Restore required image metadata from ordinary-key dictionary values.

    v1.0.0 converted Image metadata into GeoJSON Features and then read
    feature['properties']['system:index']. GeoJSON does not retain the
    system:index there. v1.0.1 never converts these records to Features.
    No identifier or acquisition date is invented and no record is skipped.
    """
    label = f'{source_collection or "scene inventory"}, record {position}'
    require(isinstance(record, dict), f'{label}: expected a metadata dictionary.')
    item = dict(record)
    index = item.pop('audit_scene_index', None)
    start_ms = item.pop('audit_time_start_ms', None)
    asset_id = item.pop('audit_image_id', None)

    require(isinstance(index, str) and bool(index.strip()),
            f'{label}: source image has no usable system:index. No row ID was substituted.')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', index) is not None,
            f'{label}: unsafe/unsupported source scene identifier {index!r}.')
    require(isinstance(start_ms, (int, float)) and not isinstance(start_ms, bool)
            and math.isfinite(float(start_ms)),
            f'{label}: source image has no finite numeric system:time_start; no date was inferred.')
    # Exercise the same conversion used downstream before accepting the record.
    try:
        datetime.fromtimestamp(float(start_ms) / 1000.0, timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f'{label}: acquisition time is outside the supported range.') from exc
    require(isinstance(asset_id, str) and bool(asset_id.strip()),
            f'{label}: source image has no usable asset identifier.')

    if source_collection is not None:
        prefix = str(source_collection).rstrip('/') + '/'
        # Image.id() may be an element ID. Qualify it only with the known
        # queried collection when it exactly equals the retrieved scene index.
        if '/' not in asset_id:
            require(asset_id == index,
                    f'{label}: image element ID conflicts with system:index.')
            asset_id = prefix + asset_id
        require(asset_id == prefix + index,
                f'{label}: image asset ID conflicts with its queried collection/index.')
    else:
        require('/' in asset_id and asset_id.rsplit('/', 1)[-1] == index,
                f'{label}: a full image asset ID matching system:index is required.')

    # Preserve the public record schema consumed by every later audit step.
    item['system:index'] = index
    item['system:time_start'] = start_ms
    item['_asset_id'] = asset_id
    return item


def ee_inventory(collection, props, maximum, source_collection=None):
    """Retrieve a bounded List of plain dictionaries, not GeoJSON Features.

    Explicit aliases preserve system properties during serialization for SR,
    cloud-probability and Cloud Score+ collections alike.
    """
    import ee
    require(isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0,
            'Inventory maximum must be a positive integer.')
    n = int(collection.size().getInfo())
    require(n <= maximum,
            f'{n} assets exceed inventory limit {maximum}; no silent truncation was applied.')
    if n == 0:
        return []

    ordinary_props = [p for p in dict.fromkeys(props) if not p.startswith('system:')]
    unsupported = [p for p in props if p.startswith('system:')
                   and p not in {'system:index', 'system:time_start'}]
    require(not unsupported, f'Unsupported system metadata selectors: {unsupported}')

    def image_dictionary(element):
        image = ee.Image(element)
        return (image.toDictionary(ordinary_props)
                .set('audit_scene_index', image.get('system:index'))
                .set('audit_time_start_ms', image.get('system:time_start'))
                .set('audit_image_id', image.id()))

    # This collection is explicitly bounded above (160 SR/480 QA assets by
    # default). One metadata-only list is fetched, not all image pixels.
    raw = collection.toList(n).map(image_dictionary).getInfo()
    require(isinstance(raw, list) and len(raw) == n,
            'Inventory response count differs from the queried collection. No records were dropped.')
    records = [normalise_scene_inventory_record(item, source_collection, i)
               for i, item in enumerate(raw)]
    require(len({r['_asset_id'] for r in records}) == n,
            'Duplicate image asset IDs in the returned scene inventory.')
    require(len({r['system:index'] for r in records}) == n,
            'Duplicate native scene indices in the returned scene inventory.')
    return records


def query_scene_inventory(cfg, times, patch_geometry, original, out):
    import ee
    from shapely.geometry import mapping
    region = ee.Geometry(mapping(patch_geometry), 'EPSG:4326', False)
    base = ee.ImageCollection(SR_COLLECTION).filterDate(times['start_utc'],times['end_exclusive_utc']).filterBounds(region)
    props = ['system:index','system:time_start','PRODUCT_ID','MGRS_TILE','SPACECRAFT_NAME',
             'SENSING_ORBIT_NUMBER','SENSING_ORBIT_DIRECTION','DATATAKE_IDENTIFIER','GENERATION_TIME',
             'PROCESSING_BASELINE','CLOUDY_PIXEL_PERCENTAGE','MEAN_SOLAR_AZIMUTH_ANGLE','MEAN_SOLAR_ZENITH_ANGLE',
             'MEAN_INCIDENCE_ZENITH_ANGLE_B8','MEAN_INCIDENCE_AZIMUTH_ANGLE_B8','GENERAL_QUALITY','GEOMETRIC_QUALITY',
             'RADIOMETRIC_QUALITY','SENSOR_QUALITY','DEGRADED_MSI_DATA_PERCENTAGE']
    records = ee_inventory(base,props,cfg['maximum_scene_assets'], source_collection=SR_COLLECTION)
    auxiliaries, aux_status = {}, []
    for key, coll in [('cloud_probability',CP_COLLECTION),('cloud_score',CS_COLLECTION)]:
        try:
            c = ee.ImageCollection(coll).filterDate(times['start_utc'],times['end_exclusive_utc']).filterBounds(region)
            auxprops = ['system:index','system:time_start','MGRS_TILE','MODEL_VERSION','NO_CONTEXT_FRACTION','SOURCE_ASSET_ID']
            ars = ee_inventory(c,auxprops,cfg['maximum_scene_assets']*3, source_collection=coll)
            grouped = {}
            for ar in ars: grouped.setdefault(str(ar['system:index']),[]).append(ar)
            auxiliaries[key] = grouped
            aux_status.append({'collection':coll,'available_assets':len(ars),'status':'QUERIED'})
        except Exception as exc:
            auxiliaries[key] = {}
            aux_status.append({'collection':coll,'status':'UNAVAILABLE_NO_PRIMARY_QA_FALLBACK','error':safe_error(exc)})
    inventory = []
    for item in records:
        item = dict(item)
        index = str(item['system:index'])
        require(re.fullmatch(r'[A-Za-z0-9_-]+', index) is not None, 'Unexpected unsafe scene identifier.')
        dt = datetime.fromtimestamp(float(item['system:time_start'])/1000,timezone.utc)
        item.update(scene_key=index, acquired_utc=iso(dt), utc_date=dt.date().isoformat(),
                    phase=classify_time(dt,cfg), days_from_event=(dt-utc(cfg['event_time_utc'])).total_seconds()/86400)
        for phase in ['PRE','POST']:
            item['listed_original_'+phase] = ('UNKNOWN_INVENTORY_MISSING' if original[phase] is None else
                                               ('YES' if index in original[phase] else 'NO'))
        for key in ['cloud_probability','cloud_score']:
            matches = auxiliaries[key].get(index,[])
            item[key+'_match_status'] = 'EXACT_INDEX' if len(matches)==1 else ('MISSING' if not matches else 'AMBIGUOUS_NOT_USED')
            item[key+'_asset_id'] = matches[0]['_asset_id'] if len(matches)==1 else None
            item[key+'_metadata'] = matches[0] if len(matches)==1 else None
        item['track_metadata_complete'] = bool(item.get('MGRS_TILE') and item.get('SPACECRAFT_NAME') and item.get('SENSING_ORBIT_NUMBER') is not None)
        item['track_key'] = '|'.join([str(item.get('SPACECRAFT_NAME','UNKNOWN')),str(item.get('SENSING_ORBIT_NUMBER','UNKNOWN')),str(item.get('MGRS_TILE','UNKNOWN'))])
        inventory.append(item)
    inventory.sort(key=lambda r:(r['acquired_utc'],r['scene_key']))
    require(len({r['scene_key'] for r in inventory})==len(inventory), 'Duplicate Sentinel-2 asset IDs.')
    write_json(out/'auxiliary_collection_status.json',aux_status)
    print(f'Scene metadata inventory PASS: {len(inventory)} SR assets; source IDs and acquisition times retained.')
    for status in aux_status:
        print('  Auxiliary inventory:', status['collection'], status['status'],
              '| assets:', status.get('available_assets', 'not available'))
    return inventory


def download_ee_image(image, ref, names, target_path, cfg):
    """Small synchronous download, fixed grid; no ee.batch.Export tasks."""
    import numpy as np
    import requests
    import rasterio
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True,exist_ok=True)
    estimated = ref['height']*ref['width']*len(names)*4
    require(estimated<28_000_000 and max(ref['height'],ref['width'])<10000,
            'Chip too large for getDownloadURL. Reduce context_margin_m.')
    errors = []
    for attempt in range(cfg['download_attempts']):
        rawpath = target_path.with_suffix('.download')
        try:
            # Default nearest-neighbour sampling onto the frozen 20 m grid.
            # New scenes have native 10/20/60 m bands; this IS explicit sampling,
            # not a claim that every input band is originally 20 m.
            url = image.toFloat().unmask(value=NODATA,sameFootprint=False).getDownloadURL({
                'name':'scene_chip','format':'GEO_TIFF','filePerBand':False,
                'crs':CRS,'crs_transform':list(ref['transform'])[:6],
                'dimensions':[ref['width'],ref['height']],
            })
            with requests.get(url,stream=True,timeout=(20,cfg['http_timeout_seconds'])) as response:
                if response.status_code != 200:
                    raise RuntimeError(f'Download returned HTTP {response.status_code}; no download URL retained.')
                with rawpath.open('wb') as f:
                    for block in response.iter_content(1024*1024):
                        if block: f.write(block)
            with rasterio.open(rawpath) as ds:
                require(ds.count==len(names) and (ds.width,ds.height)==(ref['width'],ref['height']) and str(ds.crs)==CRS,
                        'Downloaded chip schema differs from requested grid.')
                require(np.allclose(list(ds.transform)[:6],list(ref['transform'])[:6],atol=1e-6,rtol=0), 'Download transform mismatch.')
                ar = ds.read(masked=True).astype('float32').filled(NODATA)
            with rasterio.open(target_path,'w',driver='GTiff',count=len(names),height=ref['height'],width=ref['width'],
                crs=CRS,transform=ref['transform'],dtype='float32',nodata=NODATA,compress='deflate') as ds:
                ds.write(ar)
                for i,name in enumerate(names,1): ds.set_band_description(i,name)
            rawpath.unlink(missing_ok=True)
            return
        except Exception as exc:
            errors.append(safe_error(exc))
            rawpath.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            if attempt+1 < cfg['download_attempts']: time.sleep(2**attempt*2)
    raise RuntimeError('Chip download failed after retries: '+errors[-1])


def load_chip(path, names, ref):
    import numpy as np
    import rasterio
    with rasterio.open(path) as ds:
        require(list(ds.descriptions)==list(names) and str(ds.crs)==CRS and
                (ds.height,ds.width)==(ref['height'],ref['width']), 'Saved chip schema mismatch.')
        require(np.allclose(list(ds.transform)[:6],list(ref['transform'])[:6],atol=1e-6,rtol=0), 'Saved chip grid mismatch.')
        a = ds.read(masked=True).astype('float64').filled(np.nan)
    a[a==NODATA] = np.nan
    return {name:a[i] for i,name in enumerate(names)}


def terrain_chip(cfg, ref, patch_geometry, out):
    import ee
    from shapely.geometry import mapping
    path = out/'terrain_reference.tif'
    col = ee.ImageCollection(cfg['dem_collection']).filterBounds(ee.Geometry(mapping(patch_geometry)).buffer(cfg['context_margin_m']+1000))
    require(int(col.size().getInfo())>0, 'Configured DEM unavailable; no silent DEM substitution.')
    dem = col.select('DEM').mosaic().setDefaultProjection(ee.Image(col.first()).select('DEM').projection())
    terrain = ee.Terrain.products(dem)
    image = ee.Image.cat([dem.rename('elevation_m'),terrain.select('slope').rename('slope_deg'),terrain.select('aspect').rename('aspect_deg')])
    download_ee_image(image,ref,['elevation_m','slope_deg','aspect_deg'],path,cfg)
    return path


def build_scene_image(record):
    import ee
    raw = ee.Image(record['_asset_id'])
    available = raw.bandNames().getInfo()
    require(all(n in available for n in REFLECTANCE+['SCL','B8A','B9']), 'Scene lacks required reflectance/SCL/edge bands.')
    def optional(name, scale=1):
        return (raw.select(name).multiply(scale) if name in available else ee.Image.constant(NODATA)).rename(name)
    edge = raw.select('B8A').mask().And(raw.select('B9').mask()).rename('edge_valid')
    cp = (ee.Image(record['cloud_probability_asset_id']).select('probability') if record['cloud_probability_asset_id']
          else ee.Image.constant(NODATA)).rename('cloud_probability')
    cs = (ee.Image(record['cloud_score_asset_id']).select(['cs','cs_cdf']) if record['cloud_score_asset_id']
          else ee.Image.constant([NODATA,NODATA]).rename(['cs','cs_cdf']))
    return ee.Image.cat([raw.select(REFLECTANCE).multiply(0.0001),raw.select('SCL'),edge,cp,cs,
                         optional('AOT',.001),optional('WVP',.001)]).select(CHIP_BANDS)


def index_values(d):
    import numpy as np
    out = {}
    for name,a,b in [('NDVI','B8','B4'),('MNDWI','B3','B11'),('NBR','B8','B12')]:
        den = d[a]+d[b]
        ok = np.isfinite(d[a]) & np.isfinite(d[b]) & (d[a]>=0) & (d[b]>=0) & (den>1e-6)
        out[name] = np.divide(d[a]-d[b],den,out=np.full(den.shape,np.nan),where=ok)
    return out


def masks_and_metrics(d, terrain, record, cfg):
    import numpy as np
    from scipy import ndimage as ndi
    ref = np.stack([d[b] for b in REFLECTANCE])
    native = np.all(np.isfinite(ref),axis=0) & (d['edge_valid']>.5)
    scl = d['SCL']
    cp,cs = d['cloud_probability'],d['cs_cdf']
    cpok = np.isfinite(cp) & (cp>=0) & (cp<=100)
    csok = np.isfinite(cs) & (cs>=0) & (cs<=1)
    zen, az = record.get('MEAN_SOLAR_ZENITH_ANGLE'),record.get('MEAN_SOLAR_AZIMUTH_ANGLE')
    cosine = np.full(scl.shape,np.nan)
    if zen is not None and az is not None:
        z,a = math.radians(float(zen)),math.radians(float(az))
        slope,aspect = np.deg2rad(terrain['slope_deg']),np.deg2rad(terrain['aspect_deg'])
        cosine = np.cos(slope)*math.cos(z)+np.sin(slope)*math.sin(z)*np.cos(a-aspect)
    illumination = np.isfinite(cosine) & (cosine>cfg['min_cosine_illumination'])
    if zen is None or float(zen)>cfg['maximum_solar_zenith_deg']:
        illumination[:] = False
    surface = native & np.all(ref>=0,axis=0) & np.isin(scl,cfg['strict_scl_classes']) & illumination
    scl_seeds = np.isin(scl,[3,8,9,10])
    cp_seeds = cpok & (cp>=cfg['cloud_probability_max'])
    def adjacency(seeds):
        return (ndi.distance_transform_edt(~seeds,sampling=(20,20)) <= cfg['cloud_shadow_buffer_m']) if seeds.any() else np.zeros(scl.shape,bool)
    scl_near, near_cloud = adjacency(scl_seeds), adjacency(scl_seeds | cp_seeds)
    scl_base = surface & ~scl_near
    cp_base = surface & ~near_cloud
    cpclear = cpok & (cp<cfg['cloud_probability_max'])
    csclear = csok & (cs>=cfg['cloud_score_cdf_min'])
    old = native & ~np.isin(scl,[0,1,3,8,9,10]) & np.isfinite(scl) & illumination
    if record.get('cloud_probability_asset_id'): old &= cpclear
    modes = {'scl_illumination':scl_base,'cloud_probability_QA':cp_base & cpclear,
             'cloud_score_QA':scl_base & csclear,'strict_dual_QA':cp_base & cpclear & csclear,
             'legacy_like_diagnostic':old}
    metrics = {b:d[b] for b in REFLECTANCE}
    metrics.update(index_values(d))
    return native,modes,metrics,cosine,near_cloud


def summary(a):
    import numpy as np
    a = np.asarray(a,dtype=float)
    a = a[np.isfinite(a)]
    if not a.size: return {'n':0,'mean':None,'median':None,'q25':None,'q75':None,'minimum':None,'maximum':None}
    q = np.quantile(a,[.25,.5,.75],method='linear')
    return {'n':len(a),'mean':float(a.mean()),'median':float(q[1]),'q25':float(q[0]),'q75':float(q[2]),
            'minimum':float(a.min()),'maximum':float(a.max())}


def write_image(path, array, ref, names, dtype, nodata, colorinterp=None):
    import rasterio
    with rasterio.open(path,'w',driver='GTiff',height=ref['height'],width=ref['width'],count=array.shape[0],
        dtype=dtype,crs=CRS,transform=ref['transform'],nodata=nodata,compress='deflate') as ds:
        ds.write(array)
        for i,n in enumerate(names,1): ds.set_band_description(i,n)
        if colorinterp is not None: ds.colorinterp=colorinterp


def write_scene_products(scene_dir,d,native,modes,target,ref,cfg):
    import numpy as np
    from PIL import Image
    from scipy import ndimage as ndi
    from rasterio.enums import ColorInterp
    strict = np.full(target.shape,255,dtype='uint8')
    strict[native] = modes['strict_dual_QA'][native].astype('uint8')
    write_image(scene_dir/'strict_quality_support.tif',strict[None],ref,['0_rejected_1_passes_strict_QA_255_missing_SR'],'uint8',255)
    if not cfg['write_scene_RGB']: return
    rgb = np.stack([d['B4'],d['B3'],d['B2']],axis=-1)
    valid = np.all(np.isfinite(rgb),axis=2)
    display_rgb = np.clip((np.nan_to_num(rgb,nan=0)-cfg['display_min'])/(cfg['display_max']-cfg['display_min']),0,1)
    display_rgb = np.rint(display_rgb**(1/cfg['display_gamma'])*255).astype('uint8')
    rgba = np.dstack([display_rgb,valid.astype('uint8')*255])
    # Raw observation display, not cloud-masked: suspicious cloud/fog remains visible.
    write_image(scene_dir/'RGB_DISPLAY_ONLY.tif',rgba.transpose(2,0,1),ref,['RED_B4','GREEN_B3','BLUE_B2','ALPHA'],
                'uint8',None,(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha))
    boundary = target & ~ndi.binary_erosion(target,structure=ndi.generate_binary_structure(2,1),border_value=0)
    preview=rgba.copy()
    preview[boundary,:3]=255
    preview[boundary,3]=255
    Image.fromarray(preview,'RGBA').save(scene_dir/'RGB_patch_outline.png')


def scene_analysis(record,d,terrain,target,composites,cfg):
    import numpy as np
    native,modes,metrics,cosine,near = masks_and_metrics(d,terrain,record,cfg)
    n = int(target.sum())
    pct = lambda mask: 100*int(np.count_nonzero(target & mask))/n
    q = {k:record.get(k) for k in ['scene_key','acquired_utc','utc_date','phase','days_from_event','track_key',
         'SPACECRAFT_NAME','SENSING_ORBIT_NUMBER','MGRS_TILE','CLOUDY_PIXEL_PERCENTAGE','MEAN_SOLAR_ZENITH_ANGLE',
         'MEAN_SOLAR_AZIMUTH_ANGLE','cloud_probability_match_status','cloud_score_match_status','listed_original_PRE','listed_original_POST']}
    q.update(native_SR_pct_patch=pct(native),SCL_available_pct_patch=pct(np.isfinite(d['SCL'])),
        CP_available_pct_patch=pct(np.isfinite(d['cloud_probability'])),CSplus_available_pct_patch=pct(np.isfinite(d['cs_cdf'])),
        near_cloud_shadow_pct_patch=pct(near),poor_illumination_pct_patch=pct(np.isfinite(cosine)&(cosine<=cfg['min_cosine_illumination'])),
        primary_QA_pct_patch=pct(modes[cfg['primary_quality_mode']]),processing_status='SUCCESS')
    for cls in range(1,12): q[f'SCL_{cls}_pct_patch']=pct(d['SCL']==cls)
    for qa,a in [('cloud_probability',d['cloud_probability']),('cs',d['cs']),('cs_cdf',d['cs_cdf']),
                 ('cosine_illumination',cosine),('AOT',d['AOT']),('WVP',d['WVP'])]:
        z = summary(a[target]); q[qa+'_median']=z['median']; q[qa+'_available_pixels']=z['n']
    q['strict_qualified_by_coverage'] = q['primary_QA_pct_patch']>=100*cfg['minimum_patch_coverage_fraction']
    long_rows=[]
    for mode,mask in [('raw_observed_DIAGNOSTIC_ONLY',native),*modes.items()]:
        m = target & mask
        for metric,values in metrics.items():
            s=summary(values[m])
            long_rows.append({'scene_key':record['scene_key'],'acquired_utc':record['acquired_utc'],'phase':record['phase'],
                'track_key':record['track_key'],'quality_mode':mode,'metric':metric,'quality_pixels':int(m.sum()),
                'valid_pct_of_patch':100*s['n']/n,**s})
            if mode==cfg['primary_quality_mode'] and metric in ['NDVI','MNDWI','NBR']: q[metric+'_median']=s['median']
    match_rows=[]; matches={}
    for phase,a in composites.items():
        valid = target & native & np.all(np.isfinite(a),axis=0)
        equal = valid & np.all(np.abs(np.stack([d[b] for b in REFLECTANCE])-a)<=cfg['composite_match_abs_tolerance'],axis=0)
        when = utc(record['acquired_utc'])
        eligible_time = ((utc(cfg['original_primary_pre_start'])<=when<utc(cfg['event_time_utc'])) or
            (utc(cfg['original_fallback_pre_start'])<=when<utc(cfg['original_fallback_pre_end']))) if phase=='PRE' else (
            utc(cfg['event_time_utc'])<=when<utc(cfg['original_post_end_exclusive']))
        # A saved original inventory takes precedence over window assumptions.
        listed = record.get('listed_original_'+phase)
        candidate_source = (listed=='YES') if listed in ['YES','NO'] else eligible_time
        match_rows.append({'scene_key':record['scene_key'],'acquired_utc':record['acquired_utc'],'composite':phase,
            'listed_original_inventory':listed,'phase_compatible_candidate':candidate_source,
            'six_band_match_pixels':int(equal.sum()),'match_pct_patch':100*equal.sum()/n,
            'strict_QA_match_pixels':int((equal&modes['strict_dual_QA']).sum()),
            'meaning':'NUMERIC_MATCH_CANDIDATE_NOT_PROVEN_PIXEL_PROVENANCE'})
        matches[phase] = equal[target] if candidate_source else np.zeros(n,bool)
    payload = {'metrics':{k:v[target] for k,v in metrics.items()},'masks':{k:v[target] for k,v in modes.items()},
               'matches':matches,'record':record}
    return q,long_rows,match_rows,payload,(native,modes)


def date_representatives(payloads, mode, min_fraction):
    """One tile/spacecraft/orbit/date record, selected by QA count then generation and ID.
    No spectral values enter selection. All original asset results remain retained.
    """
    import numpy as np
    groups={}
    for p in payloads:
        r=p['record']
        if not r.get('track_metadata_complete'): continue
        n=int(np.count_nonzero(p['masks'][mode]))
        if n / len(p['masks'][mode]) < min_fraction: continue
        groups.setdefault((r['track_key'],r['utc_date']),[]).append(p)
    result=[]
    for _, ps in sorted(groups.items()):
        def rank(p):
            generation=p['record'].get('GENERATION_TIME') or 0
            return (-int(p['masks'][mode].sum()),-float(generation),p['record']['scene_key'])
        result.append(sorted(ps,key=rank)[0])
    return sorted(result,key=lambda p:(p['record']['acquired_utc'],p['record']['scene_key']))


def temporal_comparisons(payloads,cfg):
    """Per-track matched-pixel comparisons and a fixed common panel across dates.
    Repeat decline is descriptive. Unknown QA never becomes a clear observation.
    """
    import numpy as np
    pair_rows=[]; panel_rows=[]; persistence=[]; representative_rows=[]
    for mode in MODES[:-1]:  # legacy diagnostic excluded from persistence classification
        representatives=date_representatives(payloads,mode,cfg['minimum_patch_coverage_fraction'])
        for p in representatives:
            representative_rows.append({'quality_mode':mode,'scene_key':p['record']['scene_key'],
                'track_key':p['record']['track_key'],'utc_date':p['record']['utc_date'],
                'selection':'max_QA_coverage_then_generation_then_ID; never selected by spectral response'})
        tracks=sorted({p['record']['track_key'] for p in payloads if p['record'].get('track_metadata_complete')})
        if not tracks: tracks=['NO_TRACK_METADATA_OR_NO_SCENES']
        for track in tracks:
            group=[p for p in representatives if p['record']['track_key']==track]
            pre=[p for p in group if utc(p['record']['acquired_utc'])<utc(cfg['event_time_utc'])]
            post=[p for p in group if utc(p['record']['acquired_utc'])>=utc(cfg['event_time_utc'])]
            base=pre[-1] if pre else None
            row={'quality_mode':mode,'track_key':track,'qualified_pre_dates':len(pre),'qualified_post_dates':len(post),
                 'post_dates_after_original_window':sum(p['record']['phase']=='LATER_FOLLOWUP' for p in post),
                 'baseline_scene':base['record']['scene_key'] if base else None,'fixed_panel_pixels':0,
                 'fixed_panel_pct_patch':0.0,'median_decline_dates_on_fixed_panel':0,
                 'screen':'INSUFFICIENT_QUALITY_SCREENED_DATES','interpretation':'SPECTRAL_REPEATABILITY_ONLY_NOT_DAMAGE_VALIDATION'}
            if not base:
                row['screen']='NO_QUALITY_QUALIFIED_PRE_BASELINE'; persistence.append(row); continue
            if not post:
                row['screen']='NO_QUALITY_QUALIFIED_POST_DATE'; persistence.append(row); continue
            for p in post:
                common=base['masks'][mode] & p['masks'][mode]
                for metric in ['NDVI','MNDWI','NBR']:
                    take=common & np.isfinite(base['metrics'][metric]) & np.isfinite(p['metrics'][metric])
                    delta=p['metrics'][metric][take]-base['metrics'][metric][take]
                    s=summary(delta)
                    pair_rows.append({'quality_mode':mode,'track_key':track,'baseline_scene':base['record']['scene_key'],
                        'baseline_utc':base['record']['acquired_utc'],'post_scene':p['record']['scene_key'],'post_utc':p['record']['acquired_utc'],
                        'post_phase':p['record']['phase'],'metric':metric,'common_pixels':s['n'],
                        'common_pct_patch':100*s['n']/len(common),'sufficient_common_support':s['n']>=len(common)*cfg['minimum_patch_coverage_fraction'],
                        'median_paired_change':s['median'],'q25_paired_change':s['q25'],'q75_paired_change':s['q75'],
                        'NDVI_decline_fraction_pct':100*np.mean(delta<cfg['directional_NDVI_decline_threshold']) if metric=='NDVI' and len(delta) else None})
            # All qualified post dates included; no posthoc removal to improve persistence.
            fixed=base['masks'][mode].copy()
            for p in post: fixed &= p['masks'][mode]
            for p in [base,*post]: fixed &= np.isfinite(p['metrics']['NDVI'])
            n=int(fixed.sum()); total=len(fixed)
            row.update(fixed_panel_pixels=n,fixed_panel_pct_patch=100*n/total)
            ndvi_changes=[]
            for p in post:
                for metric in ['NDVI','MNDWI','NBR']:
                    take=fixed & np.isfinite(base['metrics'][metric]) & np.isfinite(p['metrics'][metric])
                    s=summary(p['metrics'][metric][take]-base['metrics'][metric][take])
                    panel_rows.append({'quality_mode':mode,'track_key':track,'baseline_utc':base['record']['acquired_utc'],
                        'post_utc':p['record']['acquired_utc'],'post_scene':p['record']['scene_key'],'metric':metric,
                        'fixed_common_pixels':s['n'],'median_paired_change':s['median'],
                        'post_median':summary(p['metrics'][metric][take])['median'],
                        'pre_median':summary(base['metrics'][metric][take])['median']})
                    if metric=='NDVI': ndvi_changes.append(s['median'])
            if n < cfg['minimum_patch_coverage_fraction']*total:
                row['screen']='INSUFFICIENT_FIXED_COMMON_PIXELS'
            elif len(post)<2:
                row['screen']='ONE_POST_DATE_PERSISTENCE_NOT_ASSESSABLE'
            else:
                below=[x is not None and x < cfg['directional_NDVI_decline_threshold'] for x in ndvi_changes]
                row['median_decline_dates_on_fixed_panel']=sum(below)
                if all(below):
                    row['screen']=('REPEATED_NDVI_DECLINE_IN_AVAILABLE_FOLLOWUP_DATES' if row['post_dates_after_original_window'] else
                                   'REPEATED_DECLINE_ONLY_WITHIN_ORIGINAL_POST_WINDOW')
                elif any(below): row['screen']='MIXED_AVAILABLE_DATE_TRAJECTORY'
                else: row['screen']='NO_REPEATED_MEDIAN_NDVI_DECLINE_AT_CONFIGURED_CUTOFF'
            row['first_qualified_post_utc']=post[0]['record']['acquired_utc']
            row['last_qualified_post_utc']=post[-1]['record']['acquired_utc']
            persistence.append(row)
    return pair_rows,panel_rows,persistence,representative_rows


def sync(out,drive_dir,only=None):
    if drive_dir is None: return
    paths = only if only is not None else [p for p in out.rglob('*') if p.is_file() and not p.name.endswith(('.tmp','.download'))]
    for p in paths:
        p=Path(p)
        target=drive_dir/p.relative_to(out)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and sha256(target)==sha256(p): continue
        tmp=target.with_name(target.name+'.tmp')
        shutil.copyfile(p,tmp)
        require(sha256(tmp)==sha256(p),f'Drive copy checksum failed: {target.name}')
        tmp.replace(target)


def plots_and_report(out, quality_rows, spectral_rows, inventory, cfg):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sdf=pd.DataFrame(spectral_rows)
    if not sdf.empty:
        for metric in ['NDVI','MNDWI','NBR']:
            fig,ax=plt.subplots(figsize=(10,4))
            subset=sdf[(sdf.quality_mode==cfg['primary_quality_mode'])&(sdf.metric==metric)&
                       (sdf.valid_pct_of_patch>=100*cfg['minimum_patch_coverage_fraction'])]
            for track,g in subset.groupby('track_key'):
                g=g.sort_values('acquired_utc')
                ax.plot(pd.to_datetime(g.acquired_utc,utc=True),g['median'],marker='o',linestyle='none',label=track)
            if subset.empty:
                ax.text(.5,.5,'No dates meet the primary quality/coverage screen',ha='center',transform=ax.transAxes)
            ax.axvline(utc(cfg['event_time_utc']),linestyle='--',label='Configured event time')
            ax.set(xlabel='Acquisition UTC',ylabel=metric,title=f'{cfg["patch_id"]}: strict-QA medians (date-specific support)')
            ax.text(.01,.01,'Different dates can contain different clear pixels; use fixed-panel tables for persistence.',transform=ax.transAxes,fontsize=8)
            ax.legend(fontsize=7); fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(out/f'time_series_{metric}.png',dpi=160); plt.close(fig)
    byid={r['scene_key']:r for r in quality_rows}
    blocks=[]
    for r in inventory:
        key=r['scene_key']; q=byid.get(key,{})
        png=Path('scenes')/key/'RGB_patch_outline.png'
        status=q.get('processing_status','NOT_PROCESSED')
        image=f'<img src="{html.escape(str(png))}" width="560" alt="raw unmasked scene RGB">' if (out/png).is_file() else '<p>No RGB chip available.</p>'
        blocks.append(f'<section><h2>{html.escape(r["acquired_utc"])} | {html.escape(r["phase"])}</h2>'
            f'<p>{html.escape(key)}<br>Processing: {html.escape(str(status))}; '
            f'Strict QA coverage: {html.escape(str(q.get("primary_QA_pct_patch","not available")))}%<br>'
            f'Original PRE list: {html.escape(str(r["listed_original_PRE"]))}; POST list: {html.escape(str(r["listed_original_POST"]))}</p>'
            f'{image}<p>RGB is deliberately not cloud-screened; white boundary marks the patch. '
            'Use the scene quality TIFFs/tables, not RGB brightness alone.</p></section>')
    (out/'dated_scene_review.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Dated scene audit</title>'
        '<style>body{font-family:sans-serif;max-width:1100px;margin:30px auto}section{border-top:1px solid;padding:15px 0}img{max-width:100%;image-rendering:pixelated}</style>'
        '<h1>Dated Sentinel-2 audit -- candidate, not damage validation</h1>'
        '<p>Every queried scene is listed, including cloudy, missing-QA and failed observations. '
        'The fixed display range is shared across dates. Auxiliary QA is not infallible. </p>' + ''.join(blocks)+'</html>',encoding='utf-8')



def run_inventory_fix_retry(failed_run_dir, source_code_path=None):
    """Retry a pre-inventory failure in a NEW output folder.

    Preserve the previous acquisition-time cutoff and all study settings.
    The failed folder is read-only. No image inventory had been frozen at the
    reported failure, so metadata are queried anew within the same date window.
    """
    failed = Path(failed_run_dir)
    drive_root = Path('/content/drive/MyDrive')
    if str(failed).startswith('/content/drive/') and not drive_root.is_dir():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    require(failed.is_dir(), f'Failed-run folder not found: {failed}')
    frozen_path = failed / 'run_configuration.json'
    require(frozen_path.is_file(), 'Failed run has no saved configuration; refusing to invent its settings.')
    require((failed / 'RUN_FAILED.json').is_file(), 'This helper requires an explicitly failed audit run.')
    require(not (failed / 'frozen_scene_inventory.json').exists(),
            'This run already has a frozen inventory. Use its normal resume option, not the pre-inventory retry helper.')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    require(isinstance(frozen.get('config'), dict) and isinstance(frozen.get('resolved_times'), dict),
            'The failed-run configuration has an unexpected structure.')
    saved = frozen['config']
    require(set(saved) == set(SCENE_AUDIT_CONFIG),
            'Failed-run settings differ from the v1 schema. Review them before retrying.')
    cfg = dict(saved)
    times = frozen['resolved_times']
    require(utc(times['start_utc']) == utc(cfg['review_start_utc']), 'Frozen start date/config mismatch.')
    require(utc(times['event_time_utc']) == utc(cfg['event_time_utc']), 'Frozen event date/config mismatch.')
    require(utc(times['end_exclusive_utc']) <= datetime.now(timezone.utc),
            'Frozen cutoff lies in the future; it cannot be used as an observed audit window.')
    cfg['review_end_exclusive_utc'] = times['end_exclusive_utc']
    cfg['resume_drive_run_dir'] = ''
    print('Using metadata-fix version:', VERSION)
    print('Retry source (left unchanged):', failed)
    print('Acquisition cutoff retained:', cfg['review_end_exclusive_utc'])
    print('A NEW timestamped audit folder will be created; no old outputs will be overwritten.')
    retry_record = {
        'retry_of_folder': str(failed),
        'previous_configuration_sha256': sha256(frozen_path),
        'previous_failure_sha256': sha256(failed / 'RUN_FAILED.json'),
        'previous_resolved_times': times,
        'new_code_version': VERSION,
        'prior_frozen_scene_inventory_available': False,
        'fix_scope': 'Preserve native image IDs/time through ordinary-key dictionaries, not Feature properties.',
        'acquisition_window_changed': False,
        'quality_rules_changed': False,
        'note': 'Catalog queried again now inside the retained acquisition window; prior run stopped before freezing an inventory.',
    }
    # Single-use provenance consumed by run_scene_audit, including on failure.
    global _PENDING_INVENTORY_FIX_RETRY
    _PENDING_INVENTORY_FIX_RETRY = retry_record
    try:
        return run_scene_audit(cfg, source_code_path=source_code_path)
    finally:
        _PENDING_INVENTORY_FIX_RETRY = None


def run_scene_audit(config=None, source_code_path=None):
    dependencies()
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.windows import Window
    cfg=dict(SCENE_AUDIT_CONFIG if config is None else config)
    require(cfg['primary_quality_mode']=='strict_dual_QA','Primary mode is strict_dual_QA; alternatives are labelled sensitivities.')
    require(0<cfg['minimum_patch_coverage_fraction']<=1,'Coverage fraction must lie in (0,1].')
    require(cfg['context_margin_m']>cfg['cloud_shadow_buffer_m']>=0,'Context margin must exceed cloud-adjacency buffer.')
    require(cfg['display_max']>cfg['display_min'] and cfg['display_gamma']>0,'Invalid fixed display settings.')
    needs_drive=cfg['save_to_drive'] or any(str(cfg[k]).startswith('/content/drive/') for k in ['patch_dir','s2_recovery_dir'])
    if needs_drive and not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive',force_remount=False)
    if cfg['resume_drive_run_dir']:
        drive_dir=Path(cfg['resume_drive_run_dir'])
        require((drive_dir/'run_configuration.json').is_file(), 'Resume folder has no frozen run_configuration.json.')
        frozen=json.loads((drive_dir/'run_configuration.json').read_text())
        saved=frozen['config']
        for k in cfg:
            if k not in ['resume_drive_run_dir','local_output_root','drive_output_root','save_to_drive','review_end_exclusive_utc']:
                require(cfg[k]==saved[k],f'Resume setting {k} differs from original run; start a new run instead.')
        times=frozen['resolved_times']; cfg.update(saved)
        out=Path(cfg['local_output_root'])/drive_dir.name
        if not out.exists(): shutil.copytree(drive_dir,out)
    else:
        times=resolved_times(cfg)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        name=f'{cfg["patch_id"]}_dated_scene_audit_{stamp}'
        out=Path(cfg['local_output_root'])/name
        out.mkdir(parents=True,exist_ok=False)
        drive_dir=Path(cfg['drive_output_root'])/name if cfg['save_to_drive'] else None
        if drive_dir: drive_dir.mkdir(parents=True,exist_ok=False)
    quality_rows=[]; spectral_rows=[]; match_rows=[]; payloads=[]; failures=[]
    retry_provenance = globals().get('_PENDING_INVENTORY_FIX_RETRY')
    if retry_provenance:
        write_json(out/'inventory_fix_retry.json', retry_provenance)
        sync(out,drive_dir,[out/'inventory_fix_retry.json'])
    try:
        ref,target,geog,composites,identity=load_patch_inputs(cfg,out)
        original=read_original_records(cfg,out)
        frozen={'version':VERSION,'config':cfg,'resolved_times':times,'fixed_patch_identity':identity,
                'sampling': 'New individual scenes sampled nearest-neighbour to archived 20 m grid; native band/QA resolution remains 10/20/60 m.',
                'cloud_rules':'Primary: strict SCL surface classes + edge + cosine + CP + cs_cdf + cloud/shadow adjacency; thresholds in config. Sensitivity modes remain separate.',
                'source_docs':DOCS}
        write_json(out/'run_configuration.json',frozen)
        if source_code_path and Path(source_code_path).is_file():
            shutil.copyfile(source_code_path,out/'dated_scene_audit_code_used.py')
        elif not (out/'dated_scene_audit_code_used.py').is_file():
            try:
                from IPython import get_ipython
                history=get_ipython().history_manager.input_hist_raw
                cells=[s for s in history if 'def run_scene_audit(' in s and 'LHENDE R05_P0001: DATED SENTINEL-2' in s]
                if cells: (out/'dated_scene_audit_code_used.py').write_text(cells[-1],encoding='utf-8')
            except Exception: pass
        sync(out,drive_dir)
        print('Window:',times['start_utc'],'to',times['end_exclusive_utc'],'(end exclusive, not beyond run time).')
        initialize_ee(cfg)
        invpath=out/'frozen_scene_inventory.json'
        if invpath.is_file():
            inventory=json.loads(invpath.read_text())
            print('Reusing the frozen acquisition inventory:',len(inventory),'assets.')
        else:
            inventory=query_scene_inventory(cfg,times,geog,original,out)
            write_json(invpath,inventory)
            sync(out,drive_dir)
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in inventory]).to_csv(out/'scene_inventory.csv',index=False)
        require(len(inventory)>0,'No Sentinel-2 SR assets currently available in this patch/window; no persistence conclusion possible.')
        terrain_path=out/'terrain_reference.tif'
        if not terrain_path.is_file(): terrain_chip(cfg,ref,geog,out)
        terrain=load_chip(terrain_path,['elevation_m','slope_deg','aspect_deg'],ref)
        print(f'Query returned {len(inventory)} scene assets. None excluded by whole-scene cloud percentage.')
        sync(out,drive_dir)
        for i,record in enumerate(inventory,1):
            key=record['scene_key']; scene_dir=out/'scenes'/key; scene_dir.mkdir(parents=True,exist_ok=True)
            chip=scene_dir/'SR_and_QA.tif'; receipt=scene_dir/'download_receipt.json'
            print(f'[{i}/{len(inventory)}] {record["acquired_utc"]} | {key}')
            try:
                if chip.is_file() and receipt.is_file():
                    cached=json.loads(receipt.read_text())
                    require(sha256(chip)==cached['sha256'],'Cached scene chip changed; do not silently reuse it.')
                else:
                    image=build_scene_image(record)
                    download_ee_image(image,ref,CHIP_BANDS,chip,cfg)
                    write_json(receipt,{'sha256':sha256(chip),'bytes':chip.stat().st_size,'record':record,
                        'sampled_at_utc':iso(datetime.now(timezone.utc)),'sampling':'nearest on frozen 20m grid'})
                d=load_chip(chip,CHIP_BANDS,ref)
                q,srows,mrows,payload,maskdata=scene_analysis(record,d,terrain,target,composites,cfg)
                write_scene_products(scene_dir,d,maskdata[0],maskdata[1],target,ref,cfg)
                # Exact sampled target values for subsequent paired-date comparisons.
                np.savez_compressed(scene_dir/'patch_pixel_samples.npz',**{
                    **{k:v for k,v in payload['metrics'].items()},
                    **{'mask_'+k:v for k,v in payload['masks'].items()},
                    'matches_saved_PRE':payload['matches']['PRE'],'matches_saved_POST':payload['matches']['POST']})
                write_json(scene_dir/'scene_analysis.json',{'quality':q,'spectral':srows,'composite_matches':mrows})
                (scene_dir/'SCENE_FAILED.json').unlink(missing_ok=True)
                if drive_dir is not None:
                    (drive_dir/'scenes'/key/'SCENE_FAILED.json').unlink(missing_ok=True)
                quality_rows.append(q); spectral_rows.extend(srows); match_rows.extend(mrows); payloads.append(payload)
                sync(out,drive_dir,[p for p in scene_dir.iterdir() if p.is_file()])
            except Exception as exc:
                fail={'scene_key':key,'acquired_utc':record['acquired_utc'],'phase':record['phase'],
                      'track_key':record.get('track_key'), 'processing_status':'FAILED_NOT_A_CLEAR_OR_STABLE_OBSERVATION','error':safe_error(exc)}
                quality_rows.append(fail);failures.append(fail)
                write_json(scene_dir/'SCENE_FAILED.json',fail)
                print('  Scene failed; retained as unassessable:',safe_error(exc))
                sync(out,drive_dir,[scene_dir/'SCENE_FAILED.json'])
            pd.DataFrame(quality_rows).to_csv(out/'scene_quality_summary.csv',index=False)
            pd.DataFrame(spectral_rows).to_csv(out/'scene_spectral_summaries.csv',index=False)
            pd.DataFrame(match_rows).to_csv(out/'composite_scene_numeric_matches.csv',index=False)
            sync(out,drive_dir,[out/'scene_quality_summary.csv',out/'scene_spectral_summaries.csv',out/'composite_scene_numeric_matches.csv'])
        pairs,panels,persistence,reps=temporal_comparisons(payloads,cfg)
        for row in persistence:
            row['failed_returned_assets_on_track'] = sum(f.get('track_key')==row['track_key'] for f in failures)
            if row['failed_returned_assets_on_track']:
                row['screen'] = 'PARTIAL_INPUTS__' + row['screen']
        for name,rows in [('same_track_paired_date_changes.csv',pairs),('fixed_common_pixel_panel.csv',panels),
                          ('persistence_diagnostic.csv',persistence),('selected_daily_representatives.csv',reps)]:
            pd.DataFrame(rows).to_csv(out/name,index=False)
        provenance=[]
        for phase in ['PRE','POST']:
            counts=sum((p['matches'][phase].astype('int32') for p in payloads),start=np.zeros(int(target.sum()),dtype='int32'))
            provenance.append({'composite':phase,'patch_pixels':int(target.sum()),'no_numeric_match':int((counts==0).sum()),
                'one_numeric_candidate':int((counts==1).sum()),'multiple_numeric_candidates':int((counts>1).sum()),
                'interpretation':'NOT_PROVEN_CONTRIBUTOR_DATES; match can be ambiguous and only queried archived granules were tested'})
        pd.DataFrame(provenance).to_csv(out/'composite_matching_summary.csv',index=False)
        review_form = pd.DataFrame([{
            'scene_key':r['scene_key'], 'acquired_utc':r['acquired_utc'], 'phase':r['phase'],
            'visual_review_status':'unreviewed', 'cloud_haze_or_fog':'', 'shadow_or_illumination':'',
            'misalignment_or_edge_artifact':'', 'surface_response_description':'',
            'reviewer':'', 'review_notes':'',
        } for r in inventory])
        # A blank template only: this does not overwrite any manual review decisions.
        review_form.to_csv(out/'dated_scene_visual_review_TEMPLATE.csv',index=False)
        pd.DataFrame(failures,columns=['scene_key','acquired_utc','phase','track_key','processing_status','error']).to_csv(out/'scene_failures.csv',index=False)
        plots_and_report(out,quality_rows,spectral_rows,inventory,cfg)
        manifest={'version':VERSION,'config':cfg,'resolved_times':times,'inventory_assets':len(inventory),
                  'successful_assets':len(payloads),'failed_assets':len(failures),'status':'COMPLETE' if not failures else 'PARTIAL_SCENE_FAILURES',
                  'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                    'rasterio':rasterio.__version__,'GDAL':rasterio.__gdal_version__,
                    'earthengine_api':importlib.import_module('ee').__version__,
                    'scipy':importlib.import_module('scipy').__version__,
                    'shapely':importlib.import_module('shapely').__version__,
                    'pyproj':importlib.import_module('pyproj').__version__},'limitations':[
                    'New diagnostic; original patch, M01, composite pixels and review fields remain unchanged.',
                    'The configured event date is a supplied study anchor, not independently verified by this code.',
                    'Strict QA gates and 70% coverage rule are audit settings, not validated clear-sky labels or damage thresholds.',
                    'Missing cloud-probability/Cloud Score+ never becomes a strict-clear observation; modes remain separate.',
                    'SCL and spectral QA can omit fog or flag genuinely bright surfaces; inspect unmasked dated chips.',
                    'Cosine illumination is a local DEM-slope proxy; it does not model cast shadows or recover per-pixel solar angles.',
                    'Individual new images have 10/20/60m inputs explicitly nearest-sampled to the 20m patch grid.',
                    'Same tile/orbit/spacecraft comparisons and fixed common panels reduce but do not eliminate viewing/coverage differences.',
                    'Only available dates can be examined; no future scenes or automatic subsequent monitoring.',
                    'A repeat-decline diagnostic is not a significance test, independent damage validation or proof of event causality.',
                    'Spectral matches to saved composites are candidates, never reconstructed historical pixel provenance.',
                    'Source inventories list possible contributing images, not guaranteed per-pixel contributors.',
                    'Source composites selected the target; this is a targeted observational review, not a random accuracy sample.',
                    'Apparent recovery may be transient real change or imaging effects; missing later dates are not persistence.',
                    'Failed scene downloads remain explicit and must be reviewed before interpreting date absence.'
                  ],'source_docs':DOCS}
        write_json(out/'dated_scene_audit_manifest.json',manifest)
        readme='''DATED SCENE AUDIT -- HOW TO READ\n\nPrimary mode: strict_dual_QA. Auxiliary gaps do not trigger a fallback.\nThe three other quality modes are separate sensitivity views, not interchangeable observations.\nEvery asset is inventoried; same-day duplicates are not counted as repeated dates.\nDates are grouped by spacecraft, relative orbit and MGRS tile; a most-recent qualified PRE reference is used.\nPost dates must share at least 70% of target pixels on the fixed panel for the repeatability screen.\nNo source mask is changed; unknown != stable. No validation_status is edited.\n\nStart with scene_quality_summary.csv, persistence_diagnostic.csv, and dated_scene_review.html.\nOpen the HTML after extracting the ZIP so relative scene PNG paths work.\nIn QGIS, add patch_outline.geojson above scenes/<ID>/RGB_DISPLAY_ONLY.tif.\nKeep PRE/POST review range fixed; do not use display-only bytes for numerical indices.\nThe raw per-scene RGB includes clouds intentionally. strict_quality_support.tif: 1=passes rules, 0=rejected, 255=missing SR.\nSR_and_QA.tif includes reflectance, SCL, B8A/B9 edge mask, CP, cs, cs_cdf, AOT and WVP. All numeric bands Float32; -9999 missing.\nNo quiet substitution of QA products or new scene composites.\nIf insufficient clear dates are reported, that is an audit result, not a code error.\nTo continue an interrupted download, set resume_drive_run_dir to this run's Drive folder and rerun the script.\nTo examine dates acquired later, start a NEW run with a later cutoff; keep this snapshot.\n'''
        (out/'README.txt').write_text(readme,encoding='utf-8')
        (out/'DATA_ATTRIBUTION.txt').write_text(
            'Sentinel-2: Copernicus Sentinel data. Cloud Score+: Google Earth Engine, CC-BY-4.0.\n'
            'Terrain derived using Copernicus WorldDEM-30. Produced using Copernicus WorldDEM-30 '
            '© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS '
            'by the European Union and ESA; all rights reserved.\n'
            'The organisations in charge of the Copernicus programme by law or by delegation do not incur '
            'any liability for any use of the Copernicus WorldDEM-30.\n'
            'See source_docs in dated_scene_audit_manifest.json for sources, licence links and technical documentation.\n',
            encoding='utf-8')
        pd.DataFrame([{'band':i+1,'name':n} for i,n in enumerate(CHIP_BANDS)]).to_csv(out/'scene_band_order.csv',index=False)
        files_to_hash=[p for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['output_checksums.csv','Lhende_dated_scene_audit_bundle.zip'] and not p.name.endswith(('.tmp','.download'))]
        pd.DataFrame([{'file':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files_to_hash]).to_csv(out/'output_checksums.csv',index=False)
        with zipfile.ZipFile(out/'Lhende_dated_scene_audit_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in files_to_hash+[out/'output_checksums.csv']: z.write(p,arcname=str(p.relative_to(out)))
        sync(out,drive_dir)
        print('\nDATED SCENE QUALITY -- PRIMARY STRICT QA')
        qdf=pd.DataFrame(quality_rows)
        cols=[x for x in ['acquired_utc','phase','native_SR_pct_patch','primary_QA_pct_patch','NDVI_median','MNDWI_median',
              'NBR_median','strict_qualified_by_coverage','cloud_probability_match_status','cloud_score_match_status','processing_status'] if x in qdf]
        pdf=pd.DataFrame(persistence)
        try:
            from IPython.display import display
            display(qdf[cols].round(4))
            print('\nSAME-TRACK FIXED-PIXEL PERSISTENCE DIAGNOSTIC -- NOT DAMAGE VALIDATION')
            display(pdf.loc[pdf['quality_mode'].eq(cfg['primary_quality_mode'])])
            print('\nCOMPOSITE NUMERIC-MATCH CANDIDATES -- NOT PROVEN PROVENANCE')
            display(pd.DataFrame(provenance))
        except ImportError:
            print(qdf[cols].to_string(index=False)); print(pdf.to_string(index=False))
        print('\nSaved locally:',out)
        print('Saved and checksum-verified on Drive:',drive_dir)
        print('No Earth Engine batch tasks. No old detections, geometries or review labels were changed.')
        if failures: print('REVIEW: some assets failed; see scene_failures.csv and resume instructions.')
        return {'local_dir':out,'drive_dir':drive_dir,'scene_quality':qdf,'persistence':pdf}
    except Exception as exc:
        write_json(out/'RUN_FAILED.json',{'error':safe_error(exc),'status':'AUDIT_NOT_COMPLETE','original_data_changed':False})
        sync(out,drive_dir,[out/'RUN_FAILED.json'])
        print('Audit stopped. Diagnostic folder:',drive_dir or out)
        raise


if __name__ == '__main__':
    _code = globals().get('DATED_SCENE_SCRIPT_PATH') or globals().get('__file__')
    DATED_SCENE_AUDIT_RESULTS = run_scene_audit(SCENE_AUDIT_CONFIG,source_code_path=_code)
