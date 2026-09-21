# ============================================================
# LHENDE 2026 — TOP-20 PATCH BATCH REVIEW v1.0.0
# R05 top 10 + R06 top 10 (4-neighbour primary patches)
#
# PURPOSE
#   1) Reproduce the exact candidate-patch pixels from patch_ids_4n.tif.
#   2) Characterize PRE/POST NDVI, MNDWI, NBR and stored directional change.
#   3) Reconcile saved PRE/POST composite pixels against the ORIGINAL
#      Sentinel-2 inventories, after explicit wrapper-ID normalization.
#   4) Evaluate strict dated-scene QA on numerically matching pixels.
#
# IMPORTANT
#   - No M01/M02 pixels, geometries, thresholds, or existing review labels
#     are modified.
#   - Source-scene matching is numerical compatibility with an original
#     inventory scene. It is not a claim of perfect historical provenance.
#   - Strict QA is a screening rule, not ground validation or damage proof.
#   - No automatic erosion/debris/flood/damage labels are assigned.
# ============================================================

from __future__ import annotations

import io
import os
import re
import sys
import json
import math
import time
import shutil
import hashlib
import zipfile
import warnings
import subprocess
import importlib
from pathlib import Path
from datetime import datetime, timezone

VERSION = "1.0.1"  # fixes bare Earth Engine element IDs in SR/QA scene loading
CRS = "EPSG:32645"
NODATA = -9999.0
PIXEL_M = 20.0
PIXEL_M2 = PIXEL_M * PIXEL_M

REFLECTANCE = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDICES = ["NDVI", "MNDWI", "NBR"]
CHANGE_BANDS = ["dNDVI", "dMNDWI", "dNBR", "spectral_rms"]

SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CP_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
CS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CHIP_BANDS = REFLECTANCE + ["SCL", "edge_valid", "cloud_probability", "cs_cdf"]

CONFIG = {
    # Existing completed inputs
    "patch_dir": "/content/drive/MyDrive/Lhende_2026_Candidate_Patches/M01_patch_inventory_20260911T082650777727Z",
    "s2_recovery_dir": "/content/drive/MyDrive/Lhende_2026_S2_Review/S2_export_recovery_20260911T093217241209Z",
    "original_audit_dir": "/content/drive/MyDrive/Lhende_2026_GEE_Audit",

    # Batch definition
    "reaches": ["R05", "R06"],
    "top_n_per_reach": 10,

    # Earth Engine
    "ee_project": "woven-name-441217-g5",
    "dem_collection": "COPERNICUS/DEM/GLO30_2024_1",

    # Strict QA — same primary rules used for the R05_P0001 dated-scene audit
    "strict_scl_classes": [4, 5, 6],
    "cloud_probability_max": 55.0,
    "cloud_score_cdf_min": 0.60,
    "cloud_shadow_buffer_m": 60.0,
    "min_cosine_illumination": 0.15,
    "maximum_solar_zenith_deg": 70.0,

    # Exact numerical-source reconciliation
    "composite_match_abs_tolerance": 0.000002,

    # Download extent: one small reach chip per scene, not one full corridor export
    "context_margin_m": 120.0,
    "maximum_chip_pixels": 350000,
    "download_attempts": 3,
    "http_timeout_seconds": 180,

    # Output/resume
    "output_root": "/content/drive/MyDrive/Lhende_2026_Top20_Patch_Review",
    # Leave blank for a fresh run. To resume, paste a prior output folder.
    "resume_output_dir": "",
}


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def install_if_missing(import_name: str, pip_name: str | None = None):
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])


def dependencies():
    for imp, pip in [
        ("numpy", "numpy"), ("pandas", "pandas"), ("rasterio", "rasterio"),
        ("scipy", "scipy"), ("requests", "requests"),
        ("ee", "earthengine-api"), ("openpyxl", "openpyxl"),
        ("pyproj", "pyproj"),
    ]:
        install_if_missing(imp, pip)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:1800]


def finite_summary(a):
    import numpy as np
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None,
                "iqr": None, "minimum": None, "maximum": None}
    q25, med, q75 = np.quantile(a, [0.25, 0.5, 0.75], method="linear")
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(med),
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
            "minimum": float(a.min()), "maximum": float(a.max())}


def normalize_inventory_scene_id(value: str):
    """Explicitly normalize only known Earth-Engine merge wrappers.

    Accepted payload must remain a complete Sentinel-2 granule index:
      YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_TxxYYY
    Leading numeric wrapper tokens such as 1_, 2_, 1_2_ are removed.
    Nothing is inferred from dates alone.
    """
    raw = str(value).strip().split("/")[-1]
    pattern = r"(?:\d+_)*((?:20\d{6}T\d{6})_(?:20\d{6}T\d{6})_T\d{2}[A-Z]{3})$"
    m = re.fullmatch(pattern, raw)
    if not m:
        return None, "UNRECOGNIZED_ID_FORMAT"
    canonical = m.group(1)
    prefix = raw[: -len(canonical)]
    rule = "identity" if not prefix else f"remove_numeric_wrapper:{prefix}"
    return canonical, rule


def mount_drive_if_needed():
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)


def initialize_ee(project):
    import ee
    try:
        ee.Initialize(project=project)
        ee.Number(1).getInfo()
    except Exception:
        print("Earth Engine needs authentication. Use the account authorized for the project.")
        ee.Authenticate()
        ee.Initialize(project=project)
        ee.Number(1).getInfo()
    print("Earth Engine initialized:", project)


def find_required(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct
    hits = list(root.rglob(filename))
    require(len(hits) == 1, f"Expected exactly one {filename} below {root}; found {len(hits)}")
    return hits[0]


def read_patch_inventory(patch_dir: Path):
    import pandas as pd
    geojson = find_required(patch_dir, "candidate_patches_4n_WGS84.geojson")
    obj = json.loads(geojson.read_text(encoding="utf-8-sig"))
    rows = [f["properties"] for f in obj.get("features", [])]
    frame = pd.DataFrame(rows)
    required = {"patch_id", "raster_id", "zone_id", "area_rank_in_reach", "pixel_count", "area_km2"}
    require(required.issubset(frame.columns), f"Patch inventory missing fields: {sorted(required - set(frame.columns))}")
    frame["raster_id"] = frame["raster_id"].astype(int)
    frame["area_rank_in_reach"] = frame["area_rank_in_reach"].astype(int)
    frame["pixel_count"] = frame["pixel_count"].astype(int)
    return frame, geojson


def select_top_patches(frame, cfg):
    import pandas as pd
    pieces = []
    for reach in cfg["reaches"]:
        x = frame[(frame["zone_id"] == reach) & (frame["area_rank_in_reach"] <= cfg["top_n_per_reach"])].copy()
        x = x.sort_values("area_rank_in_reach")
        require(len(x) == cfg["top_n_per_reach"],
                f"Expected {cfg['top_n_per_reach']} top patches for {reach}; found {len(x)}")
        pieces.append(x)
    return pd.concat(pieces, ignore_index=True)


def raster_reference(path: Path):
    import rasterio
    with rasterio.open(path) as ds:
        return {"width": ds.width, "height": ds.height, "transform": ds.transform,
                "crs": str(ds.crs), "nodata": ds.nodata}


def aligned_or_fail(ds, ref, label):
    import numpy as np
    require(str(ds.crs) == ref["crs"], f"{label}: CRS mismatch")
    rel = (~ds.transform) * ref["transform"]
    require(np.allclose([rel.a, rel.b, rel.d, rel.e], [1, 0, 0, 1], atol=1e-7, rtol=0),
            f"{label}: pixel scale/rotation mismatch")
    require(np.allclose([rel.c, rel.f], np.round([rel.c, rel.f]), atol=1e-6, rtol=0),
            f"{label}: grid is not integer-aligned")
    return int(round(rel.c)), int(round(rel.f))


def read_named_on_patch_grid(path: Path, names, ref):
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as ds:
        c0, r0 = aligned_or_fail(ds, ref, path.name)
        desc = list(ds.descriptions)
        indices = []
        for name in names:
            require(desc.count(name) == 1, f"{path.name}: cannot identify exactly one band named {name}")
            indices.append(desc.index(name) + 1)
        ar = ds.read(indices, window=Window(c0, r0, ref["width"], ref["height"]),
                     boundless=True, masked=True).astype("float64")
        data = ar.filled(np.nan)
        if ds.nodata is not None:
            data[data == ds.nodata] = np.nan
    return {name: data[i] for i, name in enumerate(names)}


def compute_indices_from_reflectance(d):
    import numpy as np
    out = {}
    for name, a, b in [("NDVI", "B8", "B4"), ("MNDWI", "B3", "B11"), ("NBR", "B8", "B12")]:
        den = d[a] + d[b]
        ok = np.isfinite(d[a]) & np.isfinite(d[b]) & (den != 0)
        out[name] = np.divide(d[a] - d[b], den, out=np.full(den.shape, np.nan), where=ok)
    return out


def local_patch_characterization(cfg, selected, patch_dir, s2_dir, out):
    import numpy as np
    import pandas as pd
    import rasterio

    patch_ids_path = find_required(patch_dir, "patch_ids_4n.tif")
    with rasterio.open(patch_ids_path) as ds:
        require(str(ds.crs) == CRS, "patch_ids_4n.tif must be EPSG:32645")
        require(abs(ds.transform.a - PIXEL_M) < 1e-6 and abs(ds.transform.e + PIXEL_M) < 1e-6,
                "Expected 20 m patch grid")
        labels = ds.read(1)
        ref = raster_reference(patch_ids_path)

    pre_path = find_required(s2_dir, "V04_PRE_original_export_bands_R05_R06.tif")
    post_path = find_required(s2_dir, "V05_POST_original_export_bands_R05_R06.tif")
    # R03 may be in the recovery source_inputs or original export root; find it below recovery first.
    r03_hits = list(s2_dir.rglob("R03_optical_change_metrics_RETRY.tif"))
    if not r03_hits:
        r03_hits = list(Path("/content/drive/MyDrive/Lhende_2026_GEE_Exports").glob("R03_optical_change_metrics_RETRY.tif"))
    require(len(r03_hits) == 1, f"Expected exactly one R03_optical_change_metrics_RETRY.tif; found {len(r03_hits)}")
    r03_path = r03_hits[0]

    pre_ref = read_named_on_patch_grid(pre_path, REFLECTANCE, ref)
    post_ref = read_named_on_patch_grid(post_path, REFLECTANCE, ref)
    pre_idx = compute_indices_from_reflectance(pre_ref)
    post_idx = compute_indices_from_reflectance(post_ref)
    changes = read_named_on_patch_grid(r03_path, CHANGE_BANDS, ref)

    rows = []
    for _, p in selected.iterrows():
        rid = int(p["raster_id"])
        mask = labels == rid
        n = int(mask.sum())
        require(n == int(p["pixel_count"]), f"{p['patch_id']}: raster pixel count differs from inventory")
        row = {
            "patch_id": p["patch_id"], "zone_id": p["zone_id"],
            "area_rank_in_reach": int(p["area_rank_in_reach"]),
            "raster_id": rid, "pixel_count": n, "area_km2": n * PIXEL_M2 / 1e6,
            "touches_joint_gap_8n": p.get("touches_joint_gap_8n"),
            "touches_outer_buffer_edge_8n": p.get("touches_outer_buffer_edge_8n"),
            "touches_reach_partition_8n": p.get("touches_reach_partition_8n"),
            "any_gap_or_boundary_flag": p.get("any_gap_or_boundary_flag"),
        }
        for idx in INDICES:
            a, b = finite_summary(pre_idx[idx][mask]), finite_summary(post_idx[idx][mask])
            row[f"pre_{idx}_median"] = a["median"]
            row[f"pre_{idx}_q25"] = a["q25"]
            row[f"pre_{idx}_q75"] = a["q75"]
            row[f"post_{idx}_median"] = b["median"]
            row[f"post_{idx}_q25"] = b["q25"]
            row[f"post_{idx}_q75"] = b["q75"]
        for metric in CHANGE_BANDS:
            s = finite_summary(changes[metric][mask])
            row[f"{metric}_median"] = s["median"]
            row[f"{metric}_q25"] = s["q25"]
            row[f"{metric}_q75"] = s["q75"]
            row[f"{metric}_valid_pixels"] = s["n"]
        nvalid = np.isfinite(changes["dNDVI"][mask]).sum()
        row["dNDVI_lt_m0p15_pct"] = 100 * np.mean(changes["dNDVI"][mask][np.isfinite(changes["dNDVI"][mask])] < -0.15) if nvalid else None
        row["dNDVI_le_m0p30_pct"] = 100 * np.mean(changes["dNDVI"][mask][np.isfinite(changes["dNDVI"][mask])] <= -0.30) if nvalid else None
        vals = changes["dMNDWI"][mask]; vals = vals[np.isfinite(vals)]
        row["dMNDWI_gt_p0p05_pct"] = 100 * np.mean(vals > 0.05) if len(vals) else None
        vals = changes["dNBR"][mask]; vals = vals[np.isfinite(vals)]
        row["dNBR_lt_m0p10_pct"] = 100 * np.mean(vals < -0.10) if len(vals) else None
        vals = changes["spectral_rms"][mask]; vals = vals[np.isfinite(vals)]
        row["RMS_ge_0p08_pct"] = 100 * np.mean(vals >= 0.08) if len(vals) else None

        # Descriptive spectral pattern only — not a physical-process or damage label.
        if (row["pre_NDVI_median"] is not None and row["post_NDVI_median"] is not None and
                row["pre_NDVI_median"] >= 0.35 and row["post_NDVI_median"] <= 0.15 and
                row["dNDVI_median"] is not None and row["dNDVI_median"] < -0.15):
            row["spectral_pattern_flag"] = "VEGETATION_SIGNATURE_DECLINE_PATTERN"
        else:
            row["spectral_pattern_flag"] = "OTHER_OR_MIXED_SPECTRAL_PATTERN"
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["zone_id", "area_rank_in_reach"]).reset_index(drop=True)
    frame.to_csv(out / "top20_local_spectral_characterization.csv", index=False)
    return frame, labels, ref, pre_ref, post_ref, patch_ids_path, pre_path, post_path, r03_path


def read_original_inventory(audit_dir: Path, out: Path):
    import pandas as pd
    records = []
    canonical = {"PRE": [], "POST": []}
    for phase, fn in [("PRE", "s2_pre_scene_inventory.csv"), ("POST", "s2_post_scene_inventory.csv")]:
        path = find_required(audit_dir, fn)
        df = pd.read_csv(path)
        require("system_index" in df.columns, f"{fn}: no system_index column")
        for raw in df["system_index"].dropna().astype(str):
            scene, rule = normalize_inventory_scene_id(raw)
            records.append({"phase": phase, "raw_system_index": raw, "canonical_scene_id": scene,
                            "normalization_rule": rule})
            if scene is not None:
                canonical[phase].append(scene)
        canonical[phase] = sorted(set(canonical[phase]))
    import pandas as pd
    rec = pd.DataFrame(records)
    rec.to_csv(out / "original_inventory_id_normalization.csv", index=False)
    require(canonical["PRE"] and canonical["POST"], "Original inventory normalization yielded no usable PRE or POST IDs")
    return canonical, rec


def subwindow_reference(ref, mask, margin_m):
    import numpy as np
    from rasterio.transform import Affine
    rows, cols = np.where(mask)
    require(len(rows) > 0, "Cannot create window for empty mask")
    pad = int(math.ceil(margin_m / PIXEL_M))
    r0 = max(0, int(rows.min()) - pad); r1 = min(ref["height"], int(rows.max()) + 1 + pad)
    c0 = max(0, int(cols.min()) - pad); c1 = min(ref["width"], int(cols.max()) + 1 + pad)
    t = ref["transform"] * Affine.translation(c0, r0)
    return {"row0": r0, "row1": r1, "col0": c0, "col1": c1,
            "height": r1 - r0, "width": c1 - c0, "transform": t, "crs": ref["crs"]}


def bounds_wgs84(subref):
    from pyproj import Transformer
    t = subref["transform"]
    x0, y0 = t * (0, 0)
    x1, y1 = t * (subref["width"], subref["height"])
    xmin, xmax = sorted([x0, x1]); ymin, ymax = sorted([y0, y1])
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(xmin, ymin)
    lon1, lat1 = tr.transform(xmax, ymax)
    return [min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)]


def qualify_collection_asset_id(collection_id, scene_id, reported_asset_id=None):
    """Return a collection-qualified Earth Engine asset ID with consistency checks.

    ee.Image.id() may serialize as the bare element ID (the system:index), not the
    full collection path. Passing that bare ID to ee.Image(...) makes Earth Engine
    attempt Image.load('<scene_id>'), which fails even though the image was just
    found inside the collection.  Because the record was retrieved from the
    explicitly named collection, it is safe to qualify a bare element ID with
    that exact collection path after checking that it equals system:index.
    """
    collection_id = str(collection_id).rstrip('/')
    scene_id = str(scene_id).strip()
    require(bool(collection_id) and bool(scene_id), 'Collection and scene identifiers are required.')
    prefix = collection_id + '/'

    if reported_asset_id is None or not str(reported_asset_id).strip():
        return prefix + scene_id

    reported = str(reported_asset_id).strip()
    if '/' not in reported:
        require(reported == scene_id,
                f'Bare Earth Engine element ID {reported!r} conflicts with system:index {scene_id!r}.')
        return prefix + scene_id

    require(reported.rsplit('/', 1)[-1] == scene_id,
            f'Earth Engine asset ID {reported!r} conflicts with system:index {scene_id!r}.')
    require(reported == prefix + scene_id,
            f'Asset {reported!r} was not returned from the expected collection {collection_id!r}.')
    return reported


def ee_scene_records(scene_ids, bbox, phase):
    import ee
    if not scene_ids:
        return []
    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    col = (ee.ImageCollection(SR_COLLECTION)
           .filter(ee.Filter.inList("system:index", scene_ids))
           .filterBounds(region))
    props = ["system:index", "system:time_start", "SPACECRAFT_NAME", "SENSING_ORBIT_NUMBER",
             "MGRS_TILE", "CLOUDY_PIXEL_PERCENTAGE", "MEAN_SOLAR_AZIMUTH_ANGLE",
             "MEAN_SOLAR_ZENITH_ANGLE", "GENERATION_TIME"]
    def f(im):
        im = ee.Image(im)
        # GeoJSON can omit system:* properties from Feature properties. Preserve
        # ordinary audit aliases. Image.id() may still be only the bare element ID;
        # qualify it below using the exact queried collection.
        d = im.toDictionary(props)
        d = d.set("audit_scene_index", im.get("system:index"))
        d = d.set("audit_time_start_ms", im.get("system:time_start"))
        d = d.set("audit_asset_id", im.id())
        return ee.Feature(None, d)
    data = ee.FeatureCollection(col.map(f)).getInfo()["features"]
    rows = []
    for feat in data:
        p = feat["properties"]
        sid = p.get("audit_scene_index") or Path(str(p.get("audit_asset_id", ""))).name
        require(sid in scene_ids, f"Unexpected scene returned: {sid}")
        p["scene_id"] = sid
        p["phase"] = phase
        p["asset_id"] = qualify_collection_asset_id(SR_COLLECTION, sid, p.get("audit_asset_id"))
        p["system:time_start"] = p.get("audit_time_start_ms")
        rows.append(p)
    return rows


def aux_asset_ids(scene_ids, bbox, collection_id):
    import ee
    if not scene_ids:
        return {}
    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    col = (ee.ImageCollection(collection_id)
           .filter(ee.Filter.inList("system:index", scene_ids))
           .filterBounds(region))
    def f(im):
        im = ee.Image(im)
        return ee.Feature(None, ee.Dictionary({
            "scene_id": im.get("system:index"),
            "reported_asset_id": im.id(),
        }))
    feats = ee.FeatureCollection(col.map(f)).getInfo()["features"]
    out = {}
    for x in feats:
        p = x["properties"]
        sid = p.get("scene_id")
        require(sid in scene_ids, f"Unexpected auxiliary scene returned: {sid}")
        out[sid] = qualify_collection_asset_id(collection_id, sid, p.get("reported_asset_id"))
    return out


def build_scene_image(record):
    import ee
    require(str(record.get("asset_id", "")).startswith(SR_COLLECTION + "/"),
            f"{record.get('scene_id')}: SR asset ID is not collection-qualified")
    raw = ee.Image(record["asset_id"])
    available = raw.bandNames().getInfo()
    require(all(b in available for b in REFLECTANCE + ["SCL", "B8A", "B9"]),
            f"{record['scene_id']}: missing required SR/SCL/edge bands")
    edge = raw.select("B8A").mask().And(raw.select("B9").mask()).rename("edge_valid")
    cp = (ee.Image(record["cp_asset_id"]).select("probability") if record.get("cp_asset_id")
          else ee.Image.constant(NODATA)).rename("cloud_probability")
    cs = (ee.Image(record["cs_asset_id"]).select("cs_cdf") if record.get("cs_asset_id")
          else ee.Image.constant(NODATA)).rename("cs_cdf")
    return ee.Image.cat([raw.select(REFLECTANCE).multiply(0.0001), raw.select("SCL"), edge, cp, cs]).select(CHIP_BANDS)


def download_ee_image(image, subref, names, target_path, cfg):
    import numpy as np
    import requests
    import rasterio
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = subref["height"] * subref["width"]
    require(pixels <= cfg["maximum_chip_pixels"],
            f"Requested reach chip has {pixels} pixels > configured maximum {cfg['maximum_chip_pixels']}")
    require(pixels * len(names) * 4 < 30_000_000, "Reach chip is too large for synchronous Earth Engine download")
    errs = []
    for attempt in range(cfg["download_attempts"]):
        raw = target_path.with_suffix(".download")
        try:
            url = image.toFloat().unmask(value=NODATA, sameFootprint=False).getDownloadURL({
                "name": "top20_scene_chip", "format": "GEO_TIFF", "filePerBand": False,
                "crs": CRS, "crs_transform": list(subref["transform"])[:6],
                "dimensions": [subref["width"], subref["height"]],
            })
            with requests.get(url, stream=True, timeout=(20, cfg["http_timeout_seconds"])) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                with raw.open("wb") as f:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            f.write(block)
            with rasterio.open(raw) as ds:
                require(ds.count == len(names), "Downloaded band count mismatch")
                require((ds.width, ds.height) == (subref["width"], subref["height"]), "Downloaded dimensions mismatch")
                require(str(ds.crs) == CRS, "Downloaded CRS mismatch")
                require(np.allclose(list(ds.transform)[:6], list(subref["transform"])[:6], atol=1e-6, rtol=0),
                        "Downloaded transform mismatch")
                ar = ds.read(masked=True).astype("float32").filled(NODATA)
            with rasterio.open(target_path, "w", driver="GTiff", count=len(names),
                               height=subref["height"], width=subref["width"], crs=CRS,
                               transform=subref["transform"], dtype="float32", nodata=NODATA,
                               compress="deflate") as ds:
                ds.write(ar)
                for i, n in enumerate(names, 1):
                    ds.set_band_description(i, n)
            raw.unlink(missing_ok=True)
            return
        except Exception as exc:
            errs.append(safe_error(exc)); raw.unlink(missing_ok=True); target_path.unlink(missing_ok=True)
            if attempt + 1 < cfg["download_attempts"]:
                time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"Download failed after retries: {errs[-1]}")


def load_chip(path, names, subref):
    import numpy as np
    import rasterio
    with rasterio.open(path) as ds:
        require(list(ds.descriptions) == list(names), f"{path.name}: band description mismatch")
        require((ds.height, ds.width) == (subref["height"], subref["width"]), f"{path.name}: shape mismatch")
        ar = ds.read(masked=True).astype("float64").filled(np.nan)
    ar[ar == NODATA] = np.nan
    return {n: ar[i] for i, n in enumerate(names)}


def download_terrain(subref, bbox, path, cfg):
    import ee
    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    col = ee.ImageCollection(cfg["dem_collection"]).filterBounds(region)
    require(int(col.size().getInfo()) > 0, "Configured Copernicus DEM unavailable")
    dem = col.select("DEM").mosaic().setDefaultProjection(ee.Image(col.first()).select("DEM").projection())
    terrain = ee.Terrain.products(dem)
    image = ee.Image.cat([dem.rename("elevation_m"), terrain.select("slope").rename("slope_deg"),
                          terrain.select("aspect").rename("aspect_deg")])
    download_ee_image(image, subref, ["elevation_m", "slope_deg", "aspect_deg"], path, cfg)


def strict_qa_mask(d, terrain, record, cfg):
    import numpy as np
    from scipy import ndimage as ndi
    refstack = np.stack([d[b] for b in REFLECTANCE])
    native = np.all(np.isfinite(refstack), axis=0) & (d["edge_valid"] > 0.5)
    scl = d["SCL"]; cp = d["cloud_probability"]; cs = d["cs_cdf"]
    cpok = np.isfinite(cp) & (cp >= 0) & (cp <= 100)
    csok = np.isfinite(cs) & (cs >= 0) & (cs <= 1)
    zen = record.get("MEAN_SOLAR_ZENITH_ANGLE"); az = record.get("MEAN_SOLAR_AZIMUTH_ANGLE")
    cosine = np.full(scl.shape, np.nan)
    if zen is not None and az is not None:
        z, a = math.radians(float(zen)), math.radians(float(az))
        slope = np.deg2rad(terrain["slope_deg"]); aspect = np.deg2rad(terrain["aspect_deg"])
        cosine = np.cos(slope) * math.cos(z) + np.sin(slope) * math.sin(z) * np.cos(a - aspect)
    illum = np.isfinite(cosine) & (cosine > cfg["min_cosine_illumination"])
    if zen is None or float(zen) > cfg["maximum_solar_zenith_deg"]:
        illum[:] = False
    surface = native & np.all(refstack >= 0, axis=0) & np.isin(scl, cfg["strict_scl_classes"]) & illum
    seeds = np.isin(scl, [3, 8, 9, 10]) | (cpok & (cp >= cfg["cloud_probability_max"]))
    if seeds.any():
        near = ndi.distance_transform_edt(~seeds, sampling=(PIXEL_M, PIXEL_M)) <= cfg["cloud_shadow_buffer_m"]
    else:
        near = np.zeros(scl.shape, bool)
    strict = surface & ~near & cpok & (cp < cfg["cloud_probability_max"]) & csok & (cs >= cfg["cloud_score_cdf_min"])
    return native, strict


def source_reconciliation(cfg, selected, labels, ref, pre_ref, post_ref, inventory_ids, out):
    import numpy as np
    import pandas as pd

    all_rows = []
    scene_rows = []
    failures = []

    for reach in cfg["reaches"]:
        reach_patches = selected[selected["zone_id"] == reach].copy()
        reach_ids = set(reach_patches["raster_id"].astype(int))
        reach_mask = np.isin(labels, list(reach_ids))
        subref = subwindow_reference(ref, reach_mask, cfg["context_margin_m"])
        bbox = bounds_wgs84(subref)
        r0, r1, c0, c1 = subref["row0"], subref["row1"], subref["col0"], subref["col1"]
        local_labels = labels[r0:r1, c0:c1]
        local_composites = {
            "PRE": {b: pre_ref[b][r0:r1, c0:c1] for b in REFLECTANCE},
            "POST": {b: post_ref[b][r0:r1, c0:c1] for b in REFLECTANCE},
        }

        reach_dir = out / "reach_scene_cache" / reach
        reach_dir.mkdir(parents=True, exist_ok=True)
        terrain_path = reach_dir / "terrain_reference.tif"
        if not terrain_path.is_file():
            download_terrain(subref, bbox, terrain_path, cfg)
        terrain = load_chip(terrain_path, ["elevation_m", "slope_deg", "aspect_deg"], subref)

        records = []
        for phase in ["PRE", "POST"]:
            records.extend(ee_scene_records(inventory_ids[phase], bbox, phase))
        all_scene_ids = sorted({r["scene_id"] for r in records})
        cp_map = aux_asset_ids(all_scene_ids, bbox, CP_COLLECTION)
        cs_map = aux_asset_ids(all_scene_ids, bbox, CS_COLLECTION)
        for r in records:
            r["cp_asset_id"] = cp_map.get(r["scene_id"])
            r["cs_asset_id"] = cs_map.get(r["scene_id"])

        # Store per-patch arrays of scene matches. Each entry is one bool vector over patch pixels.
        match_store = {(pid, phase): [] for pid in reach_patches["patch_id"] for phase in ["PRE", "POST"]}
        strict_store = {(pid, phase): [] for pid in reach_patches["patch_id"] for phase in ["PRE", "POST"]}
        scene_id_store = {(pid, phase): [] for pid in reach_patches["patch_id"] for phase in ["PRE", "POST"]}

        for j, record in enumerate(sorted(records, key=lambda x: (x["phase"], x.get("system:time_start", 0), x["scene_id"])), 1):
            scene_id = record["scene_id"]
            chip = reach_dir / f"{record['phase']}_{scene_id}.tif"
            print(f"{reach} scene {j}/{len(records)} | {record['phase']} | {scene_id}")
            try:
                if not chip.is_file():
                    download_ee_image(build_scene_image(record), subref, CHIP_BANDS, chip, cfg)
                d = load_chip(chip, CHIP_BANDS, subref)
                native, strict = strict_qa_mask(d, terrain, record, cfg)
                composite = local_composites[record["phase"]]
                scene_stack = np.stack([d[b] for b in REFLECTANCE])
                comp_stack = np.stack([composite[b] for b in REFLECTANCE])
                equal = native & np.all(np.isfinite(comp_stack), axis=0) & np.all(
                    np.abs(scene_stack - comp_stack) <= cfg["composite_match_abs_tolerance"], axis=0)

                for _, p in reach_patches.iterrows():
                    pid = p["patch_id"]; pmask = local_labels == int(p["raster_id"])
                    vec = equal[pmask]
                    qvec = (equal & strict)[pmask]
                    if vec.any():
                        match_store[(pid, record["phase"])].append(vec)
                        strict_store[(pid, record["phase"])].append(qvec)
                        scene_id_store[(pid, record["phase"])].append(scene_id)
                        scene_rows.append({
                            "patch_id": pid, "zone_id": reach, "phase": record["phase"], "scene_id": scene_id,
                            "spacecraft": record.get("SPACECRAFT_NAME"), "mgrs_tile": record.get("MGRS_TILE"),
                            "orbit": record.get("SENSING_ORBIT_NUMBER"),
                            "acquired_utc": (datetime.fromtimestamp(record["system:time_start"] / 1000, tz=timezone.utc).isoformat()
                                             if record.get("system:time_start") is not None else None),
                            "numeric_match_pixels": int(vec.sum()), "strict_QA_match_pixels": int(qvec.sum()),
                            "patch_pixels": int(pmask.sum()),
                            "match_pct_patch": 100 * int(vec.sum()) / int(pmask.sum()),
                            "strict_QA_match_pct_patch": 100 * int(qvec.sum()) / int(pmask.sum()),
                            "cloud_probability_available": bool(record.get("cp_asset_id")),
                            "cloud_score_available": bool(record.get("cs_asset_id")),
                            "interpretation": "ORIGINAL_INVENTORY_NUMERIC_COMPATIBILITY_NOT_GROUND_VALIDATION",
                        })
            except Exception as exc:
                failures.append({"zone_id": reach, "phase": record["phase"], "scene_id": scene_id,
                                 "error": safe_error(exc)})
                print("  scene failed and remains explicit:", safe_error(exc))

        for _, p in reach_patches.iterrows():
            pid = p["patch_id"]; n = int(p["pixel_count"])
            row = {"patch_id": pid, "zone_id": reach}
            for phase in ["PRE", "POST"]:
                mats = match_store[(pid, phase)]
                qmats = strict_store[(pid, phase)]
                if mats:
                    counts = np.sum(np.vstack(mats).astype(np.int16), axis=0)
                    qcounts = np.sum(np.vstack(qmats).astype(np.int16), axis=0)
                else:
                    counts = np.zeros(n, dtype=np.int16); qcounts = np.zeros(n, dtype=np.int16)
                matched = int((counts > 0).sum())
                one = int((counts == 1).sum()); multi = int((counts > 1).sum())
                strict_unique = int((qcounts > 0).sum())
                row.update({
                    f"{phase}_inventory_matching_scene_count": len(mats),
                    f"{phase}_matching_scene_ids": ";".join(scene_id_store[(pid, phase)]),
                    f"{phase}_unique_numeric_match_pixels": matched,
                    f"{phase}_numeric_match_coverage_pct": 100 * matched / n,
                    f"{phase}_no_numeric_match_pixels": n - matched,
                    f"{phase}_one_candidate_pixels": one,
                    f"{phase}_multiple_candidate_pixels": multi,
                    f"{phase}_strict_QA_unique_match_pixels": strict_unique,
                    f"{phase}_strict_QA_match_coverage_pct": 100 * strict_unique / n,
                })
                if matched == n and multi == 0:
                    status = "FULL_UNIQUE_NUMERIC_MATCH_TO_ORIGINAL_INVENTORY"
                elif matched == n:
                    status = "FULL_NUMERIC_MATCH_BUT_SOURCE_AMBIGUITY"
                elif matched > 0:
                    status = "PARTIAL_NUMERIC_MATCH_TO_ORIGINAL_INVENTORY"
                else:
                    status = "NO_NUMERIC_MATCH_TO_QUERIED_ORIGINAL_INVENTORY"
                row[f"{phase}_source_reconciliation_status"] = status
                row[f"{phase}_strict_QA_all_matched_pixels"] = bool(strict_unique == matched and matched > 0)
            all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    scenes = pd.DataFrame(scene_rows)
    fail = pd.DataFrame(failures)
    summary.to_csv(out / "top20_source_reconciliation_summary.csv", index=False)
    scenes.to_csv(out / "top20_source_scene_contributors.csv", index=False)
    fail.to_csv(out / "source_scene_failures.csv", index=False)
    return summary, scenes, fail


def merge_final(local_df, source_df, out):
    import pandas as pd
    final = local_df.merge(source_df, on=["patch_id", "zone_id"], how="left", validate="one_to_one")
    # Conservative machine-readable review state. This is evidence status, not damage/process validation.
    final["batch_evidence_status"] = "CHARACTERIZED_CANDIDATE"
    both = ((final["PRE_numeric_match_coverage_pct"] == 100) &
            (final["POST_numeric_match_coverage_pct"] == 100) &
            final["PRE_strict_QA_all_matched_pixels"].fillna(False) &
            final["POST_strict_QA_all_matched_pixels"].fillna(False))
    final.loc[both, "batch_evidence_status"] = "DATED_COMPOSITE_SOURCE_MATCH_AND_STRICT_QA_SUPPORTED"
    final["process_interpretation"] = "NOT_ASSIGNED_BY_BATCH_ANALYSIS"
    final["persistence_status"] = "NOT_TESTED_IN_THIS_TOP20_BATCH"
    final = final.sort_values(["zone_id", "area_rank_in_reach"]).reset_index(drop=True)
    final.to_csv(out / "TOP20_PATCH_REVIEW_MASTER.csv", index=False)
    with pd.ExcelWriter(out / "TOP20_PATCH_REVIEW_MASTER.xlsx", engine="openpyxl") as xw:
        final.to_excel(xw, sheet_name="Top20_Master", index=False)
        local_df.to_excel(xw, sheet_name="Local_Spectral", index=False)
        source_df.to_excel(xw, sheet_name="Source_Reconciliation", index=False)
    return final


def write_top20_geojson(selected, final, source_geojson, out):
    src = json.loads(Path(source_geojson).read_text(encoding="utf-8-sig"))
    props = final.set_index("patch_id").to_dict("index")
    keep = set(final["patch_id"])
    features = []
    for feat in src.get("features", []):
        pid = feat.get("properties", {}).get("patch_id")
        if pid not in keep:
            continue
        f = json.loads(json.dumps(feat))
        for k, v in props[pid].items():
            if k == "patch_id":
                continue
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            f["properties"][k] = v
        features.append(f)
    write_json(out / "TOP20_PATCH_REVIEW_WGS84.geojson", {"type": "FeatureCollection", "features": features})


def package_outputs(out, source_paths, cfg):
    import pandas as pd
    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "interpretation": "Targeted top-10-per-reach characterization; not representative accuracy sampling.",
        "limitations": [
            "Patch selection is by existing area rank, not confidence and not random validation sampling.",
            "PRE/POST indices and R03 metrics are from the same optical composites used in M01 detection.",
            "Source reconciliation is numerical compatibility with scenes in the saved original inventory; it is not ground validation.",
            "Strict QA reduces known cloud/shadow/illumination concerns but cannot prove artifact-free observations.",
            "No geomorphic process, damage, event causality, or persistence label is assigned automatically.",
            "R06 has substantially lower optical availability in the existing audit; interpret missing support explicitly.",
        ],
        "source_files": [{"path": str(p), "sha256": sha256(Path(p))} for p in source_paths if Path(p).is_file()],
    }
    write_json(out / "batch_review_manifest.json", manifest)
    files = [p for p in out.rglob("*") if p.is_file() and p.name not in ["output_checksums.csv", "Lhende_Top20_Patch_Review_Bundle.zip"]]
    pd.DataFrame([{"file": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]).to_csv(
        out / "output_checksums.csv", index=False)
    with zipfile.ZipFile(out / "Lhende_Top20_Patch_Review_Bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "Lhende_Top20_Patch_Review_Bundle.zip":
                z.write(p, arcname=str(p.relative_to(out)))


def run_batch_review(config=None, source_code_path=None):
    dependencies()
    import pandas as pd

    cfg = dict(CONFIG if config is None else config)
    mount_drive_if_needed()
    patch_dir = Path(cfg["patch_dir"]); s2_dir = Path(cfg["s2_recovery_dir"]); audit_dir = Path(cfg["original_audit_dir"])
    require(patch_dir.is_dir(), f"Patch directory not found: {patch_dir}")
    require(s2_dir.is_dir(), f"S2 recovery directory not found: {s2_dir}")
    require(audit_dir.is_dir(), f"Original audit directory not found: {audit_dir}")

    root = Path(cfg["output_root"]); root.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume_output_dir"):
        out = Path(cfg["resume_output_dir"])
        require(out.is_dir(), "resume_output_dir does not exist")
        frozen = json.loads((out / "run_configuration.json").read_text())
        for k, v in frozen["config"].items():
            if k not in ["resume_output_dir"]:
                require(cfg.get(k) == v, f"Resume config differs for {k}; start a new run")
        print("Resuming:", out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = root / f"Top20_patch_review_{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "run_configuration.json", {"version": VERSION, "config": cfg})

    inventory, source_geojson = read_patch_inventory(patch_dir)
    selected = select_top_patches(inventory, cfg)
    selected.to_csv(out / "top20_patch_selection.csv", index=False)
    print("\nTARGET PATCHES — AREA RANK, NOT CONFIDENCE")
    print(selected[["patch_id", "zone_id", "area_rank_in_reach", "pixel_count", "area_km2"]].to_string(index=False))

    print("\nStage 1/3: local spectral characterization...")
    local_df, labels, ref, pre_ref, post_ref, patch_ids_path, pre_path, post_path, r03_path = local_patch_characterization(
        cfg, selected, patch_dir, s2_dir, out)
    print("Local characterization PASS for", len(local_df), "patches.")

    inventory_ids, norm_df = read_original_inventory(audit_dir, out)
    print("Original inventory canonical IDs:", len(inventory_ids["PRE"]), "PRE;", len(inventory_ids["POST"]), "POST")

    print("\nStage 2/3: Earth Engine source-scene reconciliation and strict QA...")
    initialize_ee(cfg["ee_project"])
    source_df, contributors, failures = source_reconciliation(
        cfg, selected, labels, ref, pre_ref, post_ref, inventory_ids, out)
    if len(failures):
        print("WARNING:", len(failures), "scene/reach downloads failed. They remain explicit in source_scene_failures.csv")

    print("\nStage 3/3: merge master review table...")
    final = merge_final(local_df, source_df, out)
    write_top20_geojson(selected, final, source_geojson, out)

    source_paths = [patch_ids_path, pre_path, post_path, r03_path,
                    find_required(audit_dir, "s2_pre_scene_inventory.csv"),
                    find_required(audit_dir, "s2_post_scene_inventory.csv")]
    if source_code_path and Path(source_code_path).is_file():
        snap = out / "batch_review_code_used.py"
        shutil.copyfile(source_code_path, snap)
    package_outputs(out, source_paths, cfg)

    print("\nTOP-20 MASTER REVIEW TABLE")
    cols = ["patch_id", "area_km2", "pre_NDVI_median", "post_NDVI_median", "dNDVI_median",
            "dMNDWI_median", "dNBR_median", "spectral_rms_median",
            "PRE_numeric_match_coverage_pct", "POST_numeric_match_coverage_pct",
            "PRE_strict_QA_match_coverage_pct", "POST_strict_QA_match_coverage_pct",
            "batch_evidence_status"]
    print(final[cols].round(5).to_string(index=False))
    print("\nSaved:", out)
    print("No original detections, patch geometries, thresholds, or review labels were changed.")
    print("This targeted top-20 table is not a representative accuracy sample and does not assign damage/process labels.")
    return {"output_dir": out, "master": final, "contributors": contributors, "failures": failures}


if __name__ == "__main__":
    _source = globals().get("TOP20_BATCH_SCRIPT_PATH") or globals().get("__file__")
    TOP20_PATCH_REVIEW_RESULTS = run_batch_review(CONFIG, source_code_path=_source)
