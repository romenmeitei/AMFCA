# ============================================================
# LHENDE M01: REPRODUCIBLE R05/R06 CANDIDATE-PATCH INVENTORY v1.0
# Complete Google Colab cell / Python script.
# Reads audited M01/M02 and exact saved zones from Drive.
# No Earth Engine, reclassification, raster resampling or smoothing.
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

# --------------------------- SETTINGS ------------------------
# Use the exact successful M01 audit, NOT the background-analysis folder.
AUDIT_DIR = Path(
    '/content/drive/MyDrive/Lhende_2026_Multisensor_Concurrence/'
    'M01_audit_20260911T072517612102Z'
)
SAVE_TO_DRIVE = True
LOCAL_OUTPUT_ROOT = Path('/content/Lhende_2026_candidate_patches')
DRIVE_OUTPUT_ROOT = Path('/content/drive/MyDrive/Lhende_2026_Candidate_Patches')
INITIAL_REVIEW_PER_REACH = 10  # largest patches; NOT a confidence/sample-accuracy rank
# Primary = 4 neighbours (shared pixel edges).
# Sensitivity = 8 neighbours (edges OR corners).
# BOTH keep every candidate pixel. No minimum patch size is imposed.

# Install only missing dependencies, then record actual versions.
for package in ['numpy', 'pandas', 'scipy', 'rasterio', 'shapely', 'pyproj', 'fiona']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

import numpy as np
import pandas as pd
import scipy
from scipy import ndimage as ndi
import rasterio
from rasterio.features import rasterize, shapes
import shapely
from shapely.geometry import shape, mapping, box, MultiPolygon
from shapely.ops import transform as transform_geometry, unary_union
import pyproj
from pyproj import CRS, Transformer
import fiona

VERSION = '1.0.0'
ZONE_CODES = {'R05': 5, 'R06': 6}
GEOMETRY_VERSION = 'river_reference_candidate_v1'
SOURCE_FILES = {
    'm01': 'M01_concurrence_RMS008_audited.tif',
    'm02': 'M02_joint_optical_support.tif',
    'zones': 'river_reference_buffers_500m_used.geojson',
    'primary': 'M01_primary_R05_R06.csv',
    'coverage': 'M01_observation_coverage.csv',
    'method': 'M01_method_and_inputs.json',
}
SCHEMA = {
    'patch_id': 'str', 'raster_id': 'int', 'zone_id': 'str',
    'connectivity': 'int', 'area_rank_in_reach': 'int',
    'pixel_count': 'int', 'area_m2': 'float', 'area_ha': 'float', 'area_km2': 'float',
    'part_count': 'int', 'first_pixel_row': 'int', 'first_pixel_col': 'int',
    'centroid_easting_m': 'float', 'centroid_northing_m': 'float',
    'centroid_lon': 'float', 'centroid_lat': 'float',
    'review_easting_m': 'float', 'review_northing_m': 'float',
    'review_lon': 'float', 'review_lat': 'float',
    'adjacent_optical_gap_pixels': 'int', 'touches_optical_gap_8n': 'int',
    'adjacent_sar_gap_pixels': 'int', 'touches_sar_gap_8n': 'int',
    'adjacent_joint_gap_pixels': 'int', 'touches_joint_gap_8n': 'int',
    'touches_outer_buffer_edge_8n': 'int', 'touches_reach_partition_8n': 'int',
    'touches_raster_edge': 'int', 'any_gap_or_boundary_flag': 'int',
    'pixel_membership_sha256': 'str', 'geometry_build': 'str',
    'parent_8n_id': 'str', 'n_4n_patches_in_8n': 'int',
    'initial_review_queue': 'int', 'validation_status': 'str',
    'reviewer': 'str', 'review_date_utc': 'str', 'optical_evidence': 'str',
    'sar_evidence': 'str', 'process_interpretation': 'str',
    'evidence_reference': 'str', 'review_notes': 'str',
}


def check(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def read_mask(path):
    with rasterio.open(path) as src:
        check(src.count == 1, f'{path.name}: expected one band.')
        check(src.crs == rasterio.crs.CRS.from_epsg(32645),
              f'{path.name}: expected EPSG:32645.')
        check(src.dtypes[0] == 'uint8' and src.nodata == 255,
              f'{path.name}: expected original audited Byte raster with NoData=255.')
        check(src.scales[0] == 1 and src.offsets[0] == 0,
              f'{path.name}: unexpected scale/offset.')
        t = src.transform
        check(np.allclose([t.a, t.b, t.d, t.e], [20, 0, 0, -20], atol=1e-8, rtol=0),
              f'{path.name}: expected north-up 20 m grid.')
        check(src.width * src.height <= 25_000_000,
              'Unexpectedly large raster. Check the selected audit directory.')
        arr = src.read(1, masked=True)
        values = np.asarray(arr.data, dtype=np.uint8)
        valid = ~np.ma.getmaskarray(arr) & (values != 255)
        check(not np.any(valid & ~np.isin(values, [0, 1])),
              f'{path.name}: valid values must be 0 or 1.')
        info = {'crs': str(src.crs), 'transform': list(t)[:6],
                'width': src.width, 'height': src.height}
        return values, valid, t, info


def read_zones(path, dims, t):
    geo = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    check(geo.get('type') == 'FeatureCollection' and len(geo.get('features', [])) == 2,
          'Expected the exact two-feature R05/R06 buffer GeoJSON.')
    crs_name = geo.get('crs', {}).get('properties', {}).get('name') or 'EPSG:4326'
    projector = Transformer.from_crs(CRS.from_user_input(crs_name), 32645, always_xy=True)
    geometries = {}
    for feature in geo['features']:
        p = feature.get('properties', {})
        zid = str(p.get('zone_id', ''))
        check(zid in ZONE_CODES and zid not in geometries, 'Invalid or duplicate zone_id.')
        check(int(p.get('half_width_m', -1)) == 500 and
              p.get('geometry_version') == GEOMETRY_VERSION,
              'Wrong zone version: use the audited river-reference 500 m buffers.')
        g = shape(feature['geometry'])
        check(g.geom_type in {'Polygon', 'MultiPolygon'} and not g.is_empty and g.is_valid,
              f'{zid}: invalid source polygon; no automatic repair will be applied.')
        g = transform_geometry(projector.transform, g)
        check(g.is_valid and not g.is_empty, f'{zid}: invalid projected polygon.')
        geometries[zid] = g
    check(geometries['R05'].intersection(geometries['R06']).area <= 1.0,
          'Source R05/R06 polygons overlap by more than one square metre.')
    zones = np.zeros(dims, dtype=np.uint8)
    # EXACT upstream pixel-center rule: first R05, then only unassigned R06 pixels.
    for zid, code in ZONE_CODES.items():
        z = rasterize([(mapping(geometries[zid]), 1)], out_shape=dims,
                      transform=t, all_touched=False, fill=0, dtype='uint8') > 0
        zones[z & (zones == 0)] = code
    return zones, geometries


def exact_pixel_union(flat_indices, width, t):
    """Topology fallback: union the original square cells, without altering pixels."""
    cells = []
    for flat in flat_indices:
        row, col = divmod(int(flat), width)
        left, top = t * (col, row)
        right, bottom = t * (col + 1, row + 1)
        cells.append(box(left, bottom, right, top))
    return unary_union(cells)


def as_multi(geometry):
    check(geometry.geom_type in {'Polygon', 'MultiPolygon'}, 'Nonpolygon patch geometry.')
    return MultiPolygon([geometry]) if geometry.geom_type == 'Polygon' else geometry


def make_inventory(candidate, jointly_valid, m02, m02_valid, zones, t, connectivity,
                   review_n):
    """Label by reach, deterministically order IDs, polygonize and validate pixel identity."""
    dims = candidate.shape
    step = 1 if connectivity == 4 else 2
    connection = ndi.generate_binary_structure(2, step)
    neighbours = ndi.generate_binary_structure(2, 2)
    inside = zones > 0
    outer_edge = inside & ~ndi.binary_erosion(inside, structure=neighbours, border_value=0)
    raster_edge = np.zeros(dims, dtype=bool)
    raster_edge[[0, -1], :] = True
    raster_edge[:, [0, -1]] = True
    labels_out = np.zeros(dims, dtype=np.int32)
    records = []
    indices_by_id = {}
    next_id = 1
    pixel_m2 = abs(t.a * t.e - t.b * t.d)
    lonlat = Transformer.from_crs(32645, 4326, always_xy=True)
    grid_key = json.dumps({'transform': list(t)[:6], 'shape': list(dims),
                           'crs': 'EPSG:32645'}, sort_keys=True).encode()

    for zid, code in ZONE_CODES.items():
        z = zones == code
        selected = candidate & z
        labels, number = ndi.label(selected, structure=connection, output=np.int32)
        if number == 0:
            continue
        flats = np.flatnonzero(selected)
        labs = labels.ravel()[flats]
        sorter = np.argsort(labs, kind='stable')
        groups = np.split(flats[sorter], np.flatnonzero(np.diff(labs[sorter])) + 1)
        counts = np.bincount(labels.ravel(), minlength=number + 1)
        groups.sort(key=lambda ids: (-len(ids), int(ids[0])))

        # One-pixel, 8-neighbour flags. They do not modify any candidate pixel.
        optical_gap_near = ndi.binary_dilation(z & m02_valid & (m02 == 0),
                                               structure=neighbours)
        sar_gap_near = ndi.binary_dilation(z & ~m02_valid, structure=neighbours)
        joint_gap_near = ndi.binary_dilation(z & ~jointly_valid, structure=neighbours)
        other_reach_near = ndi.binary_dilation(inside & ~z, structure=neighbours)

        for rank, ids in enumerate(groups, start=1):
            old_label = int(labels.ravel()[ids[0]])
            check(len(ids) == int(counts[old_label]), 'Connected-component count inconsistency.')
            gid = next_id
            next_id += 1
            labels_out.ravel()[ids] = gid
            indices_by_id[gid] = ids
            patch_id = (f'{zid}_P{rank:04d}' if connectivity == 4
                        else f'{zid}_C8_{rank:04d}')
            row0, col0 = divmod(int(ids[0]), dims[1])
            gap_counts = [int(np.count_nonzero(a.ravel()[ids])) for a in
                          [optical_gap_near, sar_gap_near, joint_gap_near]]
            boundary_flags = [int(np.any(a.ravel()[ids])) for a in
                              [outer_edge, other_reach_near, raster_edge]]
            record = {
                'patch_id': patch_id, 'raster_id': gid, 'zone_id': zid,
                'connectivity': connectivity, 'area_rank_in_reach': rank,
                'pixel_count': len(ids), 'area_m2': len(ids) * pixel_m2,
                'area_ha': len(ids) * pixel_m2 / 10_000,
                'area_km2': len(ids) * pixel_m2 / 1_000_000,
                'first_pixel_row': row0, 'first_pixel_col': col0,
                'adjacent_optical_gap_pixels': gap_counts[0],
                'touches_optical_gap_8n': int(gap_counts[0] > 0),
                'adjacent_sar_gap_pixels': gap_counts[1],
                'touches_sar_gap_8n': int(gap_counts[1] > 0),
                'adjacent_joint_gap_pixels': gap_counts[2],
                'touches_joint_gap_8n': int(gap_counts[2] > 0),
                'touches_outer_buffer_edge_8n': boundary_flags[0],
                'touches_reach_partition_8n': boundary_flags[1],
                'touches_raster_edge': boundary_flags[2],
                'any_gap_or_boundary_flag': int(any(gap_counts) or any(boundary_flags)),
                'pixel_membership_sha256': hashlib.sha256(
                    grid_key + zid.encode() + np.asarray(ids, dtype='<u8').tobytes()).hexdigest(),
                'parent_8n_id': '', 'n_4n_patches_in_8n': 0,
                'initial_review_queue': int(rank <= review_n),
                'validation_status': 'unreviewed_candidate',
                'reviewer': '', 'review_date_utc': '', 'optical_evidence': '',
                'sar_evidence': '', 'process_interpretation': '',
                'evidence_reference': '', 'review_notes': '',
            }
            records.append(record)

    check(np.array_equal(labels_out > 0, candidate), 'Pixel loss/gain during patch labelling.')
    geometries = {}
    # Polygonize with shared-edge connectivity even for C8 clusters.
    # A diagonal C8 cluster can thus be a valid MultiPolygon, not an invalid bow-tie.
    for geo, gid_float in shapes(labels_out, mask=candidate, connectivity=4, transform=t):
        gid = int(gid_float)
        geometries.setdefault(gid, []).append(shape(geo))

    by_id = {}
    for record in records:
        gid = record['raster_id']
        parts = geometries.get(gid, [])
        check(bool(parts), f'No polygon for patch {gid}.')
        geometry = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        method = 'raster_polygonization'
        if not geometry.is_valid:
            # Rebuild exact cell union; no buffering, simplifying or hole filling.
            geometry = exact_pixel_union(indices_by_id[gid], dims[1], t)
            method = 'exact_pixel_union_topology_fallback'
        check(geometry.is_valid and not geometry.is_empty, f'Invalid geometry for patch {gid}.')
        geometry = as_multi(geometry)
        check(abs(geometry.area - record['area_m2']) <= 1e-6,
              f'Polygon area changed for patch {gid}.')
        centroid = geometry.centroid
        review_point = geometry.representative_point()  # guaranteed in/on this footprint
        clon, clat = lonlat.transform(centroid.x, centroid.y)
        rlon, rlat = lonlat.transform(review_point.x, review_point.y)
        record.update({
            'part_count': len(geometry.geoms), 'geometry_build': method,
            'centroid_easting_m': centroid.x, 'centroid_northing_m': centroid.y,
            'centroid_lon': clon, 'centroid_lat': clat,
            'review_easting_m': review_point.x, 'review_northing_m': review_point.y,
            'review_lon': rlon, 'review_lat': rlat,
        })
        by_id[gid] = geometry

    if records:
        back = rasterize([(mapping(by_id[r['raster_id']]), r['raster_id']) for r in records],
                         out_shape=dims, transform=t, fill=0, all_touched=False, dtype='int32')
        check(np.array_equal(back, labels_out), 'Vector-to-raster pixel/ID identity failed.')
    return labels_out, records, by_id, indices_by_id


def gpkg_records(records, geometries, point=False):
    for record in records:
        geometry = (shapely.geometry.Point(record['review_easting_m'], record['review_northing_m'])
                    if point else geometries[record['raster_id']])
        yield {'geometry': mapping(geometry), 'properties': record}


def write_gpkg_layer(path, layer_name, records, geometries, point=False):
    schema = {'geometry': 'Point' if point else 'MultiPolygon', 'properties': SCHEMA}
    with fiona.open(path, 'w', driver='GPKG', layer=layer_name, crs='EPSG:32645',
                    schema=schema, SPATIAL_INDEX='YES') as target:
        target.writerecords(gpkg_records(records, geometries, point))
    with fiona.open(path, layer=layer_name) as source:
        check(len(source) == len(records), f'GeoPackage row-count mismatch: {layer_name}')
        if not point:
            for feature in source:
                g = shape(feature['geometry'])
                check(g.is_valid and abs(g.area - feature['properties']['area_m2']) <= 1e-6,
                      f'GeoPackage geometry validation failed: {layer_name}')


def run_candidate_patch_inventory(audit_dir=AUDIT_DIR, save_to_drive=SAVE_TO_DRIVE,
                                  local_output_root=LOCAL_OUTPUT_ROOT,
                                  drive_output_root=DRIVE_OUTPUT_ROOT,
                                  review_n=INITIAL_REVIEW_PER_REACH,
                                  source_code_path=None, verbose=True):
    """Return output paths and tables. No dependency on previous live EE/Python objects."""
    audit_dir = Path(audit_dir)
    check(isinstance(review_n, int) and review_n > 0, 'review_n must be a positive integer.')
    if (save_to_drive or str(audit_dir).startswith('/content/drive/')) and not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    check(audit_dir.is_dir(), f'Audit directory not found: {audit_dir}')
    manifest_path = audit_dir / 'output_checksums.csv'
    check(manifest_path.is_file(), 'Missing original output_checksums.csv; do not mix unaudited files.')
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    check({'file', 'sha256', 'bytes'} <= set(manifest), 'Unexpected source-checksum schema.')
    check(not manifest['file'].duplicated().any(), 'Duplicate source-checksum entries.')
    manifest = manifest.set_index('file')
    input_rows = []
    for role, filename in SOURCE_FILES.items():
        p = audit_dir / filename
        check(p.is_file() and filename in manifest.index, f'Missing audited input: {filename}')
        check(p.stat().st_size == int(manifest.loc[filename, 'bytes']), f'Size mismatch: {filename}')
        digest = sha256_file(p)
        check(digest.lower() == str(manifest.loc[filename, 'sha256']).lower(),
              f'Input checksum mismatch: {filename}. Do not overwrite the original audit.')
        input_rows.append({'role': role, 'file': filename, 'source_path': str(p),
                           'bytes': p.stat().st_size, 'sha256': digest})

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(local_output_root) / ('M01_patch_inventory_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    working = out / 'source_inputs'
    working.mkdir()
    for filename in [*SOURCE_FILES.values(), 'output_checksums.csv']:
        shutil.copyfile(audit_dir / filename, working / filename)
    for item in input_rows:
        check(sha256_file(working / item['file']) == item['sha256'], 'Local input-copy checksum failed.')
    pd.DataFrame(input_rows).to_csv(out / 'input_checksums.csv', index=False)

    m01, valid1, t, info1 = read_mask(working / SOURCE_FILES['m01'])
    m02, valid2, t2, info2 = read_mask(working / SOURCE_FILES['m02'])
    check(info1 == info2 and t == t2, 'M01 and M02 must have exactly the same extent and grid.')
    zones, source_geometries = read_zones(working / SOURCE_FILES['zones'], m01.shape, t)
    inside = zones > 0
    check(not np.any((valid1 | valid2) & ~inside), 'Valid pixels extend outside the audited zones.')
    check(np.array_equal(valid1, valid2 & (m02 == 1)),
          'M01 valid support does not match M02 jointly observed pixels.')
    candidate = valid1 & (m01 == 1) & inside
    check(not np.any(candidate & ~valid2), 'Candidate pixels without SAR support.')
    pixel_km2 = abs(t.a * t.e - t.b * t.d) / 1e6

    primary = pd.read_csv(working / SOURCE_FILES['primary']).set_index('zone_id')
    coverage = pd.read_csv(working / SOURCE_FILES['coverage']).set_index('zone_id')
    check(primary.index.is_unique and set(primary.index) == set(ZONE_CODES), 'Wrong primary-table zones.')
    check(coverage.index.is_unique and set(coverage.index) == set(ZONE_CODES), 'Wrong coverage-table zones.')
    check(np.allclose(primary['rms_threshold'], 0.08, atol=1e-12, rtol=0), 'Wrong M01 threshold.')
    reproduce = []
    for zid, code in ZONE_CODES.items():
        z = zones == code
        count = int(np.count_nonzero(candidate & z))
        check(count == int(primary.loc[zid, 'concurrence_pixel_count']), f'{zid}: M01 pixel count changed.')
        measured = count * pixel_km2
        expected = float(primary.loc[zid, 'concurrence_km2'])
        check(abs(measured - expected) <= 1e-9, f'{zid}: M01 area does not reproduce.')
        for column, mask in [('buffer_grid_area_km2', z), ('sar_observed_km2', z & valid2),
                             ('jointly_observed_km2', z & valid1)]:
            area = int(np.count_nonzero(mask)) * pixel_km2
            check(abs(area - float(coverage.loc[zid, column])) <= 1e-9,
                  f'{zid}: observation/boundary mismatch in {column}.')
        reproduce.append({'zone_id': zid, 'source_candidate_pixels': count,
                          'source_area_km2': expected, 'recounted_area_km2': measured,
                          'status': 'PASS'})
    reproduction_df = pd.DataFrame(reproduce)
    if verbose:
        print('REPRODUCTION PASS: source hashes, grid, zones, masks, pixels and areas.')

    outputs = {}
    for conn in [4, 8]:
        if verbose:
            print(f'Extracting {conn}-neighbour candidate patches; preserving every pixel...')
        outputs[conn] = make_inventory(candidate, valid1, m02, valid2, zones, t, conn, review_n)
    l4, rec4, geo4, idx4 = outputs[4]
    l8, rec8, geo8, idx8 = outputs[8]
    rec8_by_id = {r['raster_id']: r for r in rec8}
    link_rows = []
    parent_counts = {}
    for record in rec4:
        parents = np.unique(l8.ravel()[idx4[record['raster_id']]])
        check(len(parents) == 1 and parents[0] > 0, '4n patch splits between 8n clusters.')
        parent = int(parents[0])
        parent_id = rec8_by_id[parent]['patch_id']
        record['parent_8n_id'] = parent_id
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        link_rows.append({'patch_id_4n': record['patch_id'], 'parent_patch_id_8n': parent_id,
                          'zone_id': record['zone_id'], 'pixels_4n': record['pixel_count']})
    for record in rec8:
        record['parent_8n_id'] = record['patch_id']
        record['n_4n_patches_in_8n'] = parent_counts.get(record['raster_id'], 0)
    parent_counts_by_name = {r['patch_id']: r['n_4n_patches_in_8n'] for r in rec8}
    for record in rec4:
        record['n_4n_patches_in_8n'] = parent_counts_by_name[record['parent_8n_id']]

    inventory4 = pd.DataFrame(rec4, columns=list(SCHEMA))
    inventory8 = pd.DataFrame(rec8, columns=list(SCHEMA))
    summary_rows = []
    size_rows = []
    for conn, records in [(4, rec4), (8, rec8)]:
        for zid in ZONE_CODES:
            subset = [r for r in records if r['zone_id'] == zid]
            pixels = sum(r['pixel_count'] for r in subset)
            check(pixels == int(primary.loc[zid, 'concurrence_pixel_count']), 'Inventory area conservation failed.')
            sizes = np.array([r['pixel_count'] for r in subset], dtype=int)
            review = [r for r in subset if r['initial_review_queue']]
            summary_rows.append({
                'zone_id': zid, 'connectivity': conn, 'patch_count': len(subset),
                'candidate_pixels': pixels, 'patch_area_km2': pixels * pixel_km2,
                'source_M01_area_km2': float(primary.loc[zid, 'concurrence_km2']),
                'single_pixel_patch_count': int(np.count_nonzero(sizes == 1)),
                'largest_patch_km2': float(sizes.max() * pixel_km2) if len(sizes) else 0.0,
                'median_patch_m2': float(np.median(sizes) * pixel_km2 * 1e6) if len(sizes) else None,
                'patches_touching_joint_gap': sum(r['touches_joint_gap_8n'] for r in subset),
                'patches_touching_outer_buffer': sum(r['touches_outer_buffer_edge_8n'] for r in subset),
                'patches_touching_reach_partition': sum(r['touches_reach_partition_8n'] for r in subset),
                'patches_with_any_gap_or_boundary': sum(r['any_gap_or_boundary_flag'] for r in subset),
                'initial_review_patch_count': len(review),
                'initial_review_area_km2': sum(r['area_km2'] for r in review),
                'status': 'PASS',
            })
            for name, low, high in [('1 pixel', 1, 1), ('2-4 pixels', 2, 4),
                                    ('5-9 pixels', 5, 9), ('10-24 pixels', 10, 24),
                                    ('25+ pixels', 25, None)]:
                selected = sizes[(sizes >= low) & ((sizes <= high) if high else True)]
                size_rows.append({'zone_id': zid, 'connectivity': conn, 'size_group': name,
                                  'patch_count': int(len(selected)), 'pixels': int(selected.sum()),
                                  'area_km2': float(selected.sum() * pixel_km2)})
    summary_df = pd.DataFrame(summary_rows)
    sensitivity = []
    for zid in ZONE_CODES:
        a = summary_df[(summary_df.zone_id == zid) & (summary_df.connectivity == 4)].iloc[0]
        b = summary_df[(summary_df.zone_id == zid) & (summary_df.connectivity == 8)].iloc[0]
        check(b.patch_count <= a.patch_count, '8n patch count cannot exceed 4n count.')
        sensitivity.append({'zone_id': zid, 'patches_4n': int(a.patch_count),
                            'patches_8n': int(b.patch_count),
                            'count_reduction_from_corner_merging': int(a.patch_count - b.patch_count),
                            'area_4n_km2': float(a.patch_area_km2), 'area_8n_km2': float(b.patch_area_km2),
                            'area_conservation': 'PASS'})

    tables = {
        'P01_candidate_patch_inventory_4n.csv': inventory4,
        'P02_connectivity_sensitivity_inventory_8n.csv': inventory8,
        'P03_patch_summary_by_reach.csv': summary_df,
        'P04_connectivity_sensitivity.csv': pd.DataFrame(sensitivity),
        'P05_patch_size_distribution.csv': pd.DataFrame(size_rows),
        'P06_4n_to_8n_patch_crosswalk.csv': pd.DataFrame(link_rows, columns=[
            'patch_id_4n', 'parent_patch_id_8n', 'zone_id', 'pixels_4n']),
        'P07_initial_review_queue_4n.csv': inventory4.loc[inventory4['initial_review_queue'] == 1].copy(),
        'P08_M01_reproduction_check.csv': reproduction_df,
    }
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False)

    # GeoTIFF ID maps retain NoData separately from observed non-candidate pixels.
    for conn, label_image in [(4, l4), (8, l8)]:
        raster = np.full(candidate.shape, -9999, dtype=np.int32)
        raster[valid1] = label_image[valid1]
        path = out / f'patch_ids_{conn}n.tif'
        with rasterio.open(path, 'w', driver='GTiff', count=1, dtype='int32',
                           width=info1['width'], height=info1['height'], crs='EPSG:32645',
                           transform=t, nodata=-9999, compress='deflate', tiled=True,
                           blockxsize=256, blockysize=256) as dst:
            dst.write(raster, 1)
            dst.set_band_description(1, f'patch_raster_id_{conn}n')
            dst.update_tags(code_version=VERSION, connectivity=str(conn),
                            interpretation='candidate_patch_not_confirmed_damage')
        with rasterio.open(path) as src:
            check(np.array_equal(src.read(1), raster), 'Saved label raster does not round-trip.')

    gpkg = out / 'Lhende_M01_candidate_patch_inventory.gpkg'
    write_gpkg_layer(gpkg, 'candidate_patches_4n', rec4, geo4)
    write_gpkg_layer(gpkg, 'sensitivity_clusters_8n', rec8, geo8)
    write_gpkg_layer(gpkg, 'review_points_4n', rec4, geo4, point=True)
    review_records = [r for r in rec4 if r['initial_review_queue']]
    write_gpkg_layer(gpkg, 'initial_review_queue_4n', review_records, geo4)
    with fiona.open(gpkg, 'w', driver='GPKG', layer='audited_river_buffers', crs='EPSG:32645',
                    schema={'geometry': 'MultiPolygon', 'properties': {'zone_id': 'str',
                            'half_width_m': 'int', 'geometry_version': 'str'}}) as dst:
        dst.writerecords([{'geometry': mapping(as_multi(g)), 'properties': {
            'zone_id': zid, 'half_width_m': 500, 'geometry_version': GEOMETRY_VERSION}}
            for zid, g in source_geometries.items()])
    expected_layers = {'candidate_patches_4n', 'sensitivity_clusters_8n', 'review_points_4n',
                       'initial_review_queue_4n', 'audited_river_buffers'}
    check(set(fiona.listlayers(gpkg)) == expected_layers, 'GeoPackage layer check failed.')

    # Portable geographic GeoJSON. Areas remain the authoritative UTM pixel-count areas.
    to_lonlat = Transformer.from_crs(32645, 4326, always_xy=True)
    geographic = [{'type': 'Feature', 'properties': r, 'geometry': mapping(transform_geometry(
        to_lonlat.transform, geo4[r['raster_id']]))} for r in rec4]
    write_json(out / 'candidate_patches_4n_WGS84.geojson',
               {'type': 'FeatureCollection', 'features': geographic})
    for filename in [SOURCE_FILES['m01'], SOURCE_FILES['m02'], SOURCE_FILES['zones']]:
        # Clear top-level QGIS entry points, in addition to immutable source_inputs copies.
        shutil.copyfile(working / filename, out / filename)

    field_notes = {
        'raster_id': 'Join to patch_ids_4n.tif or patch_ids_8n.tif, within that connectivity only.',
        'area_rank_in_reach': 'Descending pixel count, ties by first row-major pixel; not confidence.',
        'centroid_lon': 'Geometric centroid; may lie outside a concave patch.',
        'review_lon': 'Representative point inside/on the patch; preferred QGIS zoom location.',
        'pixel_membership_sha256': 'Hash of zone, grid and sorted flat indices; independent of run timestamp.',
        'adjacent_optical_gap_pixels': 'Candidate pixels with >=1 same-reach M02=0 neighbour, using 8 neighbours.',
        'adjacent_sar_gap_pixels': 'Candidate pixels with >=1 same-reach M02 NoData neighbour.',
        'adjacent_joint_gap_pixels': 'Candidate pixels bordering any same-reach unobserved M01 pixel.',
        'touches_outer_buffer_edge_8n': 'Touches rasterized boundary of combined R05/R06 support; not river bank.',
        'touches_reach_partition_8n': 'Adjacent to a pixel assigned to the other reach; possible split continuation.',
        'initial_review_queue': f'Largest {review_n} per reach, without excluding flags; targeted, not representative.',
        'geometry_build': 'Exact-pixel union used only when polygonization topology needs it; footprint must round-trip.',
        'n_4n_patches_in_8n': 'Number of shared-edge primary patches in the containing corner-connected cluster.',
        'validation_status': 'All initialized as unreviewed_candidate; no process/damage labels are inferred.',
    }
    pd.DataFrame([{'field': name, 'dtype': dtype, 'note': field_notes.get(name, '')}
                  for name, dtype in SCHEMA.items()]).to_csv(out / 'patch_field_dictionary.csv', index=False)
    pd.DataFrame([{'file': 'patch_ids_4n.tif / patch_ids_8n.tif', 'value': '-9999',
                   'meaning': 'No jointly observed support or outside buffers'},
                 {'file': 'patch_ids_4n.tif / patch_ids_8n.tif', 'value': '0',
                   'meaning': 'Jointly observed; not an M01 candidate'},
                 {'file': 'patch_ids_4n.tif / patch_ids_8n.tif', 'value': '>0',
                   'meaning': 'Patch raster_id in the corresponding inventory table'}]).to_csv(
                       out / 'patch_raster_legend.csv', index=False)

    code_sha = None
    code_snapshot = out / 'patch_inventory_code_used.py'
    if source_code_path and Path(source_code_path).is_file():
        shutil.copyfile(source_code_path, code_snapshot)
    else:
        # A pasted full Colab cell has no __file__; retain that cell from history.
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            text = ipython.history_manager.input_hist_raw[-1] if ipython else ''
            if 'def run_candidate_patch_inventory(' in text:
                code_snapshot.write_text(text, encoding='utf-8')
        except (AttributeError, IndexError):
            pass
    if code_snapshot.is_file():
        code_sha = sha256_file(code_snapshot)
    metadata = {
        'code_version': VERSION, 'code_sha256': code_sha,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_audit_dir': str(audit_dir), 'grid': info1,
        'source_files': input_rows,
        'primary_connectivity': 4, 'sensitivity_connectivity': 8,
        'reaches_labelled_separately': True, 'minimum_patch_pixels': 1,
        'resampling': 'None', 'smoothing': 'None', 'gap_filling': 'None',
        'buffer_clipping': 'Pixel-center assignment, R05 first. Vector cell footprints not re-clipped.',
        'patch_ids': 'Largest to smallest per reach; ties by first row-major pixel. Stable for unchanged inputs/config.',
        'flags': 'One-pixel 8-neighbour adjacency on original grid; overlaps between flags are possible.',
        'review_selection': f'Largest {review_n} primary patches per reach, not a statistical validation sample.',
        'tests_passed': ['source_checksum_match', 'grid_match', 'mask_semantics', 'source_area_and_coverage_reproduction',
                         '4n_and_8n_pixel_conservation', 'polygon_area_equality', 'vector_raster_identity',
                         'written_raster_identity', 'GeoPackage_count_and_geometry_checks'],
        'limitations': ['Candidate patches, not confirmed damage or causally attributed event effects.',
                       'M02 missing optical data remain unobserved, not unchanged.',
                       'Four/eight-neighbour connectivity changes patch counts, not the M01 area.',
                       'Boundary-touching patches may be truncated; a flag is not proof of error.',
                       'River geometry and snapped named endpoints are provisional.',
                       'Being inside a river buffer does not establish river-bank or channel association.',
                       'Optical and SAR windows differ; only SAR used two pre-event control intervals.',
                       'No pixel-based significance test or flood-probability estimate is produced.',
                       'Coordinate digits and pixel-edge outlines do not imply survey-level positional accuracy.'],
        'software': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'rasterio': rasterio.__version__,
                     'rasterio_GDAL': rasterio.__gdal_version__, 'shapely': shapely.__version__,
                     'GEOS': getattr(shapely, 'geos_version_string', 'unknown'),
                     'pyproj': pyproj.__version__, 'fiona': fiona.__version__},
    }
    write_json(out / 'patch_inventory_metadata.json', metadata)
    readme = f'''LHENDE M01 CANDIDATE-PATCH INVENTORY v{VERSION}

This is a candidate inventory, not damage validation. No candidate pixels were removed.
Primary connectivity: shared edges (4 neighbours). Sensitivity: edges or corners (8).
Every component is labelled separately within R05 or R06; shared reach-edge pixels go to R05.
Polygon area = original 20 m pixel count x 400 m2. No smoothing, vector simplification,
convex hulls, hole filling or interpolation was used. Complete edge cells are retained,
so vector footprints can extend slightly past the source polygon boundary. This matches
pixel-centre assignment; do not clip those vectors again and expect unchanged areas.
Invalid polygonization topology, when present, is rebuilt as an exact union of its cells,
recorded per patch, and verified by area and pixel-by-pixel rasterization.

OPEN IN QGIS
1. Add Lhende_M01_candidate_patch_inventory.gpkg.
2. Start with candidate_patches_4n, review_points_4n and audited_river_buffers.
3. Keep the original audited M01 and M02 TIFFs for pixel/support comparison.
4. Use initial_review_queue_4n for the largest {review_n} patches per reach.
   Size order is a navigation aid, NOT confidence or a representative validation sample.
5. Zoom using review_points_4n; geometric centroids can lie outside concave polygons.
6. Make a working copy before editing validation_status, reviewer, review_date_utc,
   optical_evidence, sar_evidence, process_interpretation, evidence_reference, review_notes.
   All are initially unreviewed or empty. Do not auto-label patches as damage.

FLAGS
Adjacency includes immediate diagonal neighbours (up to 28.3 m centre spacing).
Optical gap: M02=0 in same reach. SAR gap: M02 NoData in same reach.
Joint gap: M01 unobserved in same reach. Outer-buffer edge and inter-reach boundary
are separate flags. The latter can split a physical feature. Flags may overlap;
never sum their areas as disjoint classes. They do not delete or downgrade patches.

REPRODUCIBILITY
P08 must reproduce the source M01 CSV exactly at pixel level.
P03/P04 must conserve M01 area under both connectivities. Four- and eight-neighbour
inventories describe the SAME pixels and must not be summed together.
patch_ids_4n.tif: -9999 unknown/outside, 0 observed noncandidate, positive raster_id.
IDs are reproducible for identical rasters, zones and settings, not across changed inputs.
The source_inputs folder contains exact copied audited inputs and the upstream checksum list.
output_checksums.csv covers all output files except itself and the ZIP; the ZIP excludes itself.
A timestamped output folder protects earlier runs. Checksums test file integrity, not scientific accuracy.
'''
    (out / 'README.txt').write_text(readme, encoding='utf-8')
    output_files = sorted(p for p in out.rglob('*') if p.is_file())
    checksum_df = pd.DataFrame([{'file': p.relative_to(out).as_posix(),
                                'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in output_files])
    checksum_df.to_csv(out / 'output_checksums.csv', index=False)
    zip_path = out / 'Lhende_M01_candidate_patch_inventory_bundle.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p != zip_path:
                archive.write(p, arcname=p.relative_to(out).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        check(archive.testzip() is None, 'ZIP integrity check failed.')
    destination = None
    if save_to_drive:
        destination = Path(drive_output_root) / out.name
        destination.mkdir(parents=True, exist_ok=False)
        for p in sorted(out.rglob('*')):
            if p.is_file():
                target = destination / p.relative_to(out)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
                check(sha256_file(target) == sha256_file(p), f'Drive copy verification failed: {target.name}')
    if verbose:
        try:
            from IPython.display import display
        except ImportError:
            display = print
        print('\nPRIMARY FOUR-NEIGHBOUR PATCH SUMMARY')
        display(summary_df[summary_df.connectivity == 4].round(6))
        print('\nCONNECTIVITY SENSITIVITY — SAME PIXELS, DIFFERENT GROUPING')
        display(pd.DataFrame(sensitivity).round(6))
        print('\nINITIAL TARGETED REVIEW QUEUE — SIZE ORDER, NOT CONFIDENCE')
        display(tables['P07_initial_review_queue_4n.csv'][[
            'patch_id', 'zone_id', 'pixel_count', 'area_km2', 'review_lon', 'review_lat',
            'touches_joint_gap_8n', 'touches_outer_buffer_edge_8n', 'touches_reach_partition_8n']])
        print('\nALL CONSERVATION AND OUTPUT CHECKS PASSED.')
        print('Local output:', out)
        print('Drive output:', destination if destination else 'not requested')
        print('Open in QGIS: Lhende_M01_candidate_patch_inventory.gpkg')
        print('No old files were changed. No Earth Engine tasks were submitted.')
    return {'local_dir': out, 'drive_dir': destination, 'summary': summary_df,
            'inventory_4n': inventory4, 'inventory_8n': inventory8,
            'connectivity_sensitivity': pd.DataFrame(sensitivity), 'reproduction': reproduction_df}


if __name__ == '__main__':
    M01_PATCH_RESULTS = run_candidate_patch_inventory(
        source_code_path=globals().get('M01_PATCH_SCRIPT_PATH') or globals().get('__file__')
    )
