"""Non-destructive, offline reconciliation of the completed R05_P0001 audit.

No Earth Engine calls, new scenes, threshold changes, or edits to source outputs.
Checks saved inventory-ID formatting and recomputes six-band matching unions
from checksum-verified cached scenes. Numerical matching is NOT definitive
historical source-pixel provenance and does not establish persistence.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys

CONFIG = {
    'audit_dir': '/content/drive/MyDrive/Lhende_2026_Dated_Scene_Audit/R05_P0001_dated_scene_audit_20260911T113248637214Z',
    'local_output_root': '/content/Lhende_2026_Inventory_Reconciliation',
    'drive_output_root': '/content/drive/MyDrive/Lhende_2026_Dated_Scene_Audit',
    'save_to_drive': True,
}
BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
CANONICAL = r'\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3}'
# Deliberately narrow: keep BOTH timestamps and the exact MGRS tile.
ID_PATTERN = re.compile(r'(?P<prefix>(?:[12]_)*)(?P<scene>' + CANONICAL + r')')
ASSET_ROOT = 'COPERNICUS/S2_SR_HARMONIZED/'


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalise_inventory_id(value):
    """Return (canonical key or None, explicit transformation).

    Only whitespace, the exact known collection path and leading 1_/2_
    wrappers are recognized. Unknown wrappers stay unresolved; no fuzzy,
    timestamp-only, partial-tile or product-generation substitution occurs.
    """
    raw = str(value)
    text = raw.strip()
    rules = ['trim_whitespace'] if text != raw else []
    if text.startswith(ASSET_ROOT):
        text = text[len(ASSET_ROOT):]
        rules.append('remove_known_collection_path')
    m = ID_PATTERN.fullmatch(text)
    if m is None:
        return None, 'UNRECOGNIZED_FORMAT'
    if m.group('prefix'):
        rules.append('remove_numeric_wrapper:' + m.group('prefix'))
    return m.group('scene'), ';'.join(rules) or 'UNCHANGED'


def as_utc(value):
    x = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    need(x.tzinfo is not None, 'Timestamp lacks timezone: ' + str(value))
    return x.astimezone(timezone.utc)


def in_phase_window(acquired, phase, cfg):
    when, event = as_utc(acquired), as_utc(cfg['event_time_utc'])
    if phase == 'PRE':
        return ((as_utc(cfg['original_primary_pre_start']) <= when < event) or
                (as_utc(cfg['original_fallback_pre_start']) <= when <
                 as_utc(cfg['original_fallback_pre_end'])))
    return event <= when < as_utc(cfg['original_post_end_exclusive'])


def run_inventory_reconciliation(config=None, source_code_path=None):
    cfg = dict(CONFIG if config is None else config)
    for mod in ['numpy', 'pandas', 'rasterio']:
        if importlib.util.find_spec(mod) is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', mod])
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.windows import Window

    root = Path(cfg['audit_dir'])
    if str(root).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    need(root.is_dir(), f'Completed audit folder missing: {root}')
    original_hashes = root / 'output_checksums.csv'
    need(original_hashes.is_file(), 'Completed audit has no output_checksums.csv.')
    manifest = pd.read_csv(original_hashes, keep_default_na=False)
    need({'file', 'sha256', 'bytes'} <= set(manifest.columns), 'Unexpected checksum schema.')
    need(not manifest['file'].duplicated().any(), 'Duplicate source-checksum entries.')
    manifest = manifest.set_index('file')
    accessed = {}

    def checked(rel):
        rel = str(rel)
        p = Path(rel)
        need(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative path.')
        if rel in accessed:
            return root / rel
        need(rel in manifest.index, 'Source file lacks a recorded checksum: ' + rel)
        source = root / rel
        need(source.is_file(), 'Saved file is missing: ' + str(source))
        h = digest(source)
        need(source.stat().st_size == int(manifest.loc[rel, 'bytes']) and
             h == str(manifest.loc[rel, 'sha256']).lower(),
             'Saved source was modified: ' + rel)
        accessed[rel] = {'file': rel, 'source_path': str(source), 'sha256': h,
                         'bytes': source.stat().st_size}
        return source

    frozen = json.loads(checked('run_configuration.json').read_text())
    science = frozen['config']
    scene_records = json.loads(checked('frozen_scene_inventory.json').read_text())
    old_matches = pd.read_csv(checked('composite_scene_numeric_matches.csv'), keep_default_na=False)
    need(not old_matches.duplicated(['composite', 'scene_key']).any(), 'Duplicate saved match records.')
    old_lookup = old_matches.set_index(['composite', 'scene_key'])
    expected_pairs = {(phase, str(r['scene_key'])) for phase in ['PRE', 'POST'] for r in scene_records}
    need(set(old_lookup.index) == expected_pairs, 'Saved matching rows do not cover the frozen inventory.')
    need(len({r['scene_key'] for r in scene_records}) == len(scene_records), 'Duplicate frozen scene IDs.')

    id_rows, phase_lists = [], {}
    for phase, name in [('PRE', 's2_pre_scene_inventory.csv'), ('POST', 's2_post_scene_inventory.csv')]:
        table = pd.read_csv(checked('source_inputs/original_inventory/' + name),
                            dtype=str, keep_default_na=False)
        need('system_index' in table.columns, 'Original inventory lacks system_index: ' + name)
        rows = []
        for row_number, rec in enumerate(table.to_dict('records')):
            raw = rec['system_index']
            canonical, rule = normalise_inventory_id(raw)
            tile = str(rec.get('mgrs_tile', '')).strip().upper()
            tile_conflict = bool(canonical and tile and tile != canonical.split('_T')[-1])
            row = {'phase': phase, 'inventory_row_zero_based': row_number,
                   'raw_system_index': raw, 'canonical_scene_id': canonical or '',
                   'normalization_rule': rule, 'inventory_mgrs_tile': tile,
                   'tile_conflict': tile_conflict,
                   'usable_identifier': canonical is not None and not tile_conflict,
                   'product_id': rec.get('product_id', '')}
            rows.append(row); id_rows.append(row)
        phase_lists[phase] = rows
    normalized_df = pd.DataFrame(id_rows)

    label_path = checked('source_inputs/patch/patch_ids_4n.tif')
    identity = json.loads(checked('patch_identity.json').read_text())
    patch_id = str(science['patch_id'])
    need(identity['patch_id'] == patch_id and int(identity['connectivity']) == 4, 'Patch identity differs.')
    with rasterio.open(label_path) as ds:
        full = ds.read(1, masked=True)
        need(ds.count == 1 and ds.crs.to_epsg() == 32645, 'Wrong patch ID raster.')
        full_transform, full_crs = ds.transform, ds.crs
        need(np.allclose([full_transform.a, full_transform.b, full_transform.d, full_transform.e],
                         [20, 0, 0, -20], atol=1e-9, rtol=0), 'Wrong patch grid.')
        patch_full = (~np.ma.getmaskarray(full)) & (full.data == int(identity['raster_id']))
        flat = np.flatnonzero(patch_full)
        need(len(flat) == int(identity['pixel_count']), 'Patch pixel count changed.')
        grid_key = json.dumps({'transform': list(full_transform)[:6], 'shape': list(full.shape),
                               'crs': 'EPSG:32645'}, sort_keys=True).encode()
        member_hash = hashlib.sha256(grid_key + str(identity['zone_id']).encode()
                                     + np.asarray(flat, dtype='<u8').tobytes()).hexdigest()
        need(member_hash == identity['pixel_membership_sha256'], 'Patch membership changed.')
    n = len(flat)
    need(n > 0, 'Empty patch.')
    r, c = np.where(patch_full)
    r0, r1, c0, c1 = int(r.min()), int(r.max()+1), int(c.min()), int(c.max()+1)
    target = patch_full[r0:r1, c0:c1]
    ref_transform = rasterio.windows.transform(Window(c0, r0, c1-c0, r1-r0), full_transform)

    def aligned_band_values(path, names):
        with rasterio.open(path) as ds:
            relative = (~ds.transform) * ref_transform
            need(ds.crs == full_crs and np.allclose(
                [relative.a, relative.b, relative.d, relative.e], [1, 0, 0, 1], atol=1e-7, rtol=0)
                and np.allclose([relative.c, relative.f], np.round([relative.c, relative.f]),
                                atol=1e-6, rtol=0), 'Source grid differs: ' + str(path))
            descriptions = list(ds.descriptions)
            need(all(descriptions.count(name) == 1 for name in names),
                 'Missing/ambiguous band names: ' + str(path))
            indices = [descriptions.index(name)+1 for name in names]
            need(all(ds.scales[b-1] == 1 and ds.offsets[b-1] == 0 for b in indices),
                 'Unexpected band scaling: ' + str(path))
            win = Window(int(round(relative.c)), int(round(relative.f)), target.shape[1], target.shape[0])
            arr = ds.read(indices, window=win, boundless=True, masked=True).astype('float64')
            values = arr.filled(np.nan)
            values[values == -9999] = np.nan
            return {name: values[i] for i, name in enumerate(names)}

    composites = {}
    for phase, name in [('PRE', 'V04_PRE_original_export_bands_R05_R06.tif'),
                        ('POST', 'V05_POST_original_export_bands_R05_R06.tif')]:
        values = aligned_band_values(checked('source_inputs/composites/' + name), BANDS)
        composites[phase] = np.stack([values[b][target] for b in BANDS])
        need(np.isfinite(composites[phase]).all(), 'Missing composite target values.')

    tolerance = float(science['composite_match_abs_tolerance'])
    need(np.isfinite(tolerance) and tolerance > 0, 'Invalid frozen numerical tolerance.')
    cases = ['raw_any_queried_scene', 'phase_window_only_not_inventory_proof',
             'normalized_inventory_and_phase_window']
    counters = {(p, case): np.zeros(n, dtype='int32') for p in ['PRE', 'POST'] for case in cases}
    strict_counters = {key: np.zeros(n, dtype='int32') for key in counters}
    detailed, scene_members = [], []
    for record in scene_records:
        scene = str(record['scene_key'])
        need(re.fullmatch(CANONICAL, scene) is not None, 'Unexpected frozen scene-key format.')
        values = aligned_band_values(checked('scenes/' + scene + '/SR_and_QA.tif'), BANDS + ['edge_valid'])
        sr = np.stack([values[b][target] for b in BANDS])
        native = np.isfinite(sr).all(axis=0) & (values['edge_valid'][target] > 0.5)
        with np.load(checked('scenes/' + scene + '/patch_pixel_samples.npz'), allow_pickle=False) as npz:
            strict = np.array(npz['mask_strict_dual_QA'], dtype=bool)
        need(strict.shape == (n,) and not np.any(strict & ~native), 'Strict-mask/sample consistency failure.')
        for phase in ['PRE', 'POST']:
            match = native & np.all(np.abs(sr - composites[phase]) <= tolerance, axis=0)
            observed_n = int(match.sum())
            strict_n = int((match & strict).sum())
            old = old_lookup.loc[(phase, scene)]
            need(observed_n == int(old['six_band_match_pixels']) and
                 strict_n == int(old['strict_QA_match_pixels']),
                 'Recomputed raw matches differ from saved audit: ' + phase + '/' + scene)
            linked = [x for x in phase_lists[phase]
                      if x['usable_identifier'] and x['canonical_scene_id'] == scene]
            raw_linked = [x for x in linked if x['raw_system_index'] == scene]
            time_ok = in_phase_window(record['acquired_utc'], phase, science)
            listed = bool(linked)
            status = ('EXACT_RAW_ID_MATCH' if raw_linked else
                      'MATCH_AFTER_EXPLICIT_ID_NORMALIZATION' if linked else
                      'NOT_FOUND_AFTER_SAFE_NORMALIZATION')
            detailed.append({'composite': phase, 'scene_key': scene,
                'acquired_utc': record['acquired_utc'],
                'listed_original_inventory_saved': old['listed_original_inventory'],
                'new_identifier_status': status, 'normalized_inventory_rows': len(linked),
                'original_ids_found': json.dumps([x['raw_system_index'] for x in linked]),
                'within_frozen_phase_window': time_ok,
                'normalized_inventory_and_time_eligible': listed and time_ok,
                'six_band_match_pixels': observed_n, 'strict_QA_match_pixels': strict_n,
                'raw_match_reproduction': 'PASS'})
            use = [True, time_ok, listed and time_ok]
            for case, eligible in zip(cases, use):
                if eligible:
                    counters[(phase, case)] += match
                    strict_counters[(phase, case)] += match & strict
            if observed_n:
                for k in np.flatnonzero(match):
                    scene_members.append({'composite': phase, 'scene_key': scene,
                        'acquired_utc': record['acquired_utc'], 'patch_flat_index': int(flat[k]),
                        'passes_strict_QA': bool(strict[k]), 'within_phase_window': time_ok,
                        'listed_after_safe_normalization': listed})

    summary = []
    for (phase, case), counts in counters.items():
        scounts = strict_counters[(phase, case)]
        summary.append({'composite': phase, 'candidate_rule': case, 'patch_pixels': n,
            'unique_pixels_with_match': int((counts > 0).sum()),
            'pixels_with_no_match': int((counts == 0).sum()),
            'pixels_with_one_candidate': int((counts == 1).sum()),
            'pixels_with_multiple_candidates': int((counts > 1).sum()),
            'unique_pixels_with_strict_QA_match': int((scounts > 0).sum()),
            'match_coverage_pct': float(100*np.mean(counts > 0)),
            'interpretation': 'NUMERIC_COMPATIBILITY_NOT_PROVEN_HISTORICAL_PROVENANCE'})
    detail_df, union_df = pd.DataFrame(detailed), pd.DataFrame(summary)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(cfg['local_output_root']) / (patch_id + '_inventory_reconciliation_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    tables = {'inventory_ID_mapping.csv': normalized_df,
              'reconciled_scene_eligibility.csv': detail_df,
              'pixel_union_matching_summary.csv': union_df,
              'matched_pixel_candidates.csv': pd.DataFrame(scene_members, columns=[
                  'composite','scene_key','acquired_utc','patch_flat_index','passes_strict_QA',
                  'within_phase_window','listed_after_safe_normalization']),
              'verified_source_files.csv': pd.DataFrame(accessed.values())}
    for filename, table in tables.items():
        table.to_csv(out / filename, index=False)
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'version': '1.0',
        'source_audit': str(root), 'source_checksum_manifest_sha256': digest(original_hashes),
        'patch_id': patch_id, 'patch_pixels': n, 'membership_sha256': member_hash,
        'six_band_absolute_match_tolerance': tolerance, 'thresholds_changed': False,
        'original_data_changed': False, 'new_satellite_requests': False,
        'phase_window_settings': {k: science[k] for k in [
            'event_time_utc','original_primary_pre_start','original_fallback_pre_start',
            'original_fallback_pre_end','original_post_end_exclusive']},
        'normalization_rule': 'Trim whitespace, remove exact collection path and only leading 1_/2_ tokens; retain both timestamps and full tile.',
        'raw_count_reproduction': 'PASS_FOR_ALL_RECORDS',
        'limitations': ['Matching is numerical compatibility, not definitive scene provenance.',
            'PRE scene counts cannot be summed without checking the unique-pixel union.',
            'Inventory rows which cannot be normalized remain unresolved.',
            'No persistence result or review label is changed by metadata reconciliation.']}
    (out/'reconciliation_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if source_code_path and Path(source_code_path).is_file():
        shutil.copyfile(source_code_path, out/'reconciliation_code_used.py')
    files = sorted(p for p in out.iterdir() if p.is_file())
    pd.DataFrame([{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]).to_csv(
        out/'output_checksums.csv',index=False)
    destination = None
    if cfg['save_to_drive']:
        destination = Path(cfg['drive_output_root']) / out.name
        destination.mkdir(parents=True, exist_ok=False)
        for p in sorted(out.iterdir()):
            if p.is_file():
                q = destination/p.name
                shutil.copyfile(p,q)
                need(digest(q) == digest(p), 'Output Drive copy checksum failure.')
    try:
        from IPython.display import display
    except ImportError:
        display = print
    print('\nREPRODUCTION PASS: patch and every original raw scene-match count.')
    print('\nRAW INVENTORY ID FORMATS — original strings preserved')
    display(normalized_df[['phase','raw_system_index','canonical_scene_id','normalization_rule','tile_conflict']])
    print('\nSCENES WITH NUMERICAL MATCHES — METADATA RECONCILIATION')
    display(detail_df.loc[detail_df.six_band_match_pixels > 0, [
        'composite','scene_key','new_identifier_status','original_ids_found',
        'within_frozen_phase_window','six_band_match_pixels','strict_QA_match_pixels']])
    print('\nUNIQUE-PIXEL MATCH COVERAGE — NOT A PERSISTENCE TEST')
    display(union_df)
    print('\nSaved locally:', out)
    print('Saved to Drive:', destination)
    print('No inputs, date cutoffs, masks, tolerances, review labels or persistence conclusions changed.')
    return {'local_dir':out,'drive_dir':destination,'inventory_ID_mapping':normalized_df,
            'reconciliation':detail_df,'pixel_union_summary':union_df}


if __name__ == '__main__':
    INVENTORY_RECONCILIATION_RESULTS = run_inventory_reconciliation(
        CONFIG, source_code_path=globals().get('__file__'))
