# ============================================================
# LHENDE 2026 — PRIORITY-PATCH EXTERNAL CORROBORATION + FIGURES v1.0.0
#
# PURPOSE
#   1) Freeze the completed Top-20 Sentinel-2/SAR review as the input.
#   2) Select patches meeting the pre-defined batch evidence condition:
#        DATED_COMPOSITE_SOURCE_MATCH_AND_STRICT_QA_SUPPORTED
#   3) Query independent Copernicus EMS Rapid Mapping activation EMSR927
#      and measure AOI / mapped-vector intersections with those patches.
#   4) Query independent Landsat Collection-2 L2 observations and, where
#      possible, make same-spacecraft + same-WRS-track PRE/POST comparisons.
#   5) Generate publication-quality descriptive R05-vs-R06 figures from
#      the frozen Top-20 table.
#
# IMPORTANT INTERPRETATION LIMITS
#   - CEMS intersection means a patch intersects an official mapped vector
#     layer; preserve the product/layer attributes before inferring damage type.
#   - No CEMS intersection is not evidence of no impact unless the patch lies
#     inside a mapped AOI/product and the relevant crisis layer was available.
#   - Landsat spectral agreement is independent-sensor corroboration of a
#     spectral direction, not proof of event causality or process.
#   - Small R06 patches may contain only a few 30-m Landsat pixels. Pixel count
#     and common-clear support are therefore reported explicitly.
#   - No original detection, geometry, threshold, validation status, or review
#     label is modified by this workflow.
# ============================================================

from __future__ import annotations

import os
import re
import io
import sys
import json
import math
import time
import shutil
import hashlib
import zipfile
import subprocess
import importlib
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

VERSION = "1.0.0"
EVENT_TIME_UTC = "2026-08-26T02:52:10Z"  # frozen study setting
CRS_UTM = "EPSG:32645"
CRS_WGS84 = "EPSG:4326"

CONFIG = {
    # Frozen successful Top-20 result supplied by the user.
    "top20_run_dir": (
        "/content/drive/MyDrive/Lhende_2026_Top20_Patch_Review/"
        "Top20_patch_review_20260917T100013355438Z"
    ),

    # Priority definition. This is a technical evidence condition, not a damage rank.
    "priority_status": "DATED_COMPOSITE_SOURCE_MATCH_AND_STRICT_QA_SUPPORTED",

    # Output
    "output_root": "/content/drive/MyDrive/Lhende_2026_External_Corroboration",
    "resume_output_dir": "",

    # Copernicus EMS Rapid Mapping
    "cems_activation_code": "EMSR927",
    "cems_public_api": (
        "https://rapidmapping.emergency.copernicus.eu/"
        "backend/dashboard-api/public-activations-info/"
    ),
    "cems_request_timeout_seconds": 90,
    "cems_max_pages": 30,
    "cems_page_size": 100,
    "cems_max_geojson_bytes": 80_000_000,
    # Only used to identify potentially crisis-relevant layer names. Full layer
    # names and attributes remain in the outputs, so nothing is discarded.
    "cems_relevant_keywords": [
        "event", "extent", "flood", "affected", "damage", "damaged",
        "destroy", "grading", "debris", "mudflow", "landslide",
        "transport", "building", "infrastructure"
    ],

    # Earth Engine / Landsat independent-sensor check
    "ee_project": "woven-name-441217-g5",
    "landsat_collections": [
        "LANDSAT/LC08/C02/T1_L2",
        "LANDSAT/LC09/C02/T1_L2",
    ],
    "landsat_start": "2026-08-01T00:00:00Z",
    "landsat_end_exclusive": "2026-09-18T00:00:00Z",
    "landsat_min_clear_fraction": 0.70,
    "landsat_min_valid_pixels": 3,
    "landsat_scale_m": 30,
    "landsat_max_scene_assets": 120,
    "landsat_common_clear_min_fraction": 0.50,

    # Figure settings. No custom colors are set; default matplotlib colors are used.
    "figure_dpi": 300,
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
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("requests", "requests"),
        ("shapely", "shapely"),
        ("pyproj", "pyproj"),
        ("ee", "earthengine-api"),
        ("openpyxl", "openpyxl"),
    ]:
        install_if_missing(imp, pip)


def mount_drive_if_needed():
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:2500]


def utc(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso(dt):
    return utc(dt).isoformat().replace("+00:00", "Z")


def resolve_output(cfg):
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    if cfg.get("resume_output_dir"):
        out = Path(cfg["resume_output_dir"])
        require(out.is_dir(), f"Resume directory not found: {out}")
        return out
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / f"Priority5_external_corroboration_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def load_frozen_top20(cfg, out):
    import pandas as pd
    from shapely.geometry import shape

    run = Path(cfg["top20_run_dir"])
    master_path = run / "TOP20_PATCH_REVIEW_MASTER.csv"
    geo_path = run / "TOP20_PATCH_REVIEW_WGS84.geojson"
    require(master_path.is_file(), f"Missing Top-20 master: {master_path}")
    require(geo_path.is_file(), f"Missing Top-20 geometry: {geo_path}")

    master = pd.read_csv(master_path)
    required = {
        "patch_id", "zone_id", "area_rank_in_reach", "pixel_count", "area_km2",
        "pre_NDVI_median", "post_NDVI_median", "dNDVI_median",
        "dMNDWI_median", "dNBR_median", "spectral_rms_median",
        "PRE_numeric_match_coverage_pct", "POST_numeric_match_coverage_pct",
        "PRE_strict_QA_match_coverage_pct", "POST_strict_QA_match_coverage_pct",
        "batch_evidence_status"
    }
    missing = sorted(required - set(master.columns))
    require(not missing, f"Top-20 master missing required columns: {missing}")
    require(len(master) == 20, f"Expected 20 Top-20 rows; found {len(master)}")
    require(master["patch_id"].is_unique, "Top-20 master patch IDs are not unique")

    obj = json.loads(geo_path.read_text(encoding="utf-8"))
    geom_by_id = {}
    props_by_id = {}
    for feat in obj.get("features", []):
        pid = str(feat.get("properties", {}).get("patch_id", "")).strip()
        if pid:
            require(pid not in geom_by_id, f"Duplicate patch geometry: {pid}")
            geom_by_id[pid] = shape(feat["geometry"])
            props_by_id[pid] = dict(feat.get("properties", {}))
    require(set(master.patch_id) <= set(geom_by_id), "Top-20 GeoJSON lacks one or more master-table patches")

    priority = master.loc[master["batch_evidence_status"] == cfg["priority_status"]].copy()
    require(len(priority) > 0, f"No patches have priority status {cfg['priority_status']!r}")
    priority = priority.sort_values(["zone_id", "area_rank_in_reach", "patch_id"]).reset_index(drop=True)

    # Freeze the exact successful inputs inside the new output provenance.
    input_rows = []
    for p, role in [(master_path, "top20_master"), (geo_path, "top20_geometry")]:
        input_rows.append({"role": role, "path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(input_rows).to_csv(out / "input_checksums.csv", index=False)
    priority.to_csv(out / "priority_patch_selection.csv", index=False)

    print("\nPRIORITY PATCHES — TECHNICAL EVIDENCE CONDITION, NOT DAMAGE RANK")
    print(priority[["patch_id", "zone_id", "area_rank_in_reach", "pixel_count", "area_km2",
                    "dNDVI_median", "dMNDWI_median", "dNBR_median"]].to_string(index=False))
    return master, priority, geom_by_id, props_by_id


# -----------------------------------------------------------------------------
# Publication-quality descriptive figures from the frozen Top-20 result
# -----------------------------------------------------------------------------

def make_figures(master, out, cfg):
    import numpy as np
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": cfg["figure_dpi"],
        "savefig.dpi": cfg["figure_dpi"],
    })

    # Figure 1: directional optical response. One axes only, no subplot.
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for reach, marker in [("R05", "o"), ("R06", "^")]:
        d = master.loc[master.zone_id == reach]
        sizes = 45 + 500 * np.sqrt(d.area_km2 / master.area_km2.max())
        ax.scatter(d.dNDVI_median, d.dMNDWI_median, s=sizes, marker=marker,
                   alpha=0.78, label=reach)
        for _, r in d.iterrows():
            ax.annotate(r.patch_id.replace(reach + "_", ""),
                        (r.dNDVI_median, r.dMNDWI_median),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axvline(0, linewidth=0.8, linestyle="--")
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Median dNDVI (POST − PRE)")
    ax.set_ylabel("Median dMNDWI (POST − PRE)")
    ax.set_title("Top-10 candidate patches per reach: directional optical response")
    ax.legend(title="Reach")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"Figure1_Top20_dNDVI_vs_dMNDWI.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: NBR versus NDVI directional response.
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for reach, marker in [("R05", "o"), ("R06", "^")]:
        d = master.loc[master.zone_id == reach]
        sizes = 45 + 500 * np.sqrt(d.area_km2 / master.area_km2.max())
        ax.scatter(d.dNDVI_median, d.dNBR_median, s=sizes, marker=marker,
                   alpha=0.78, label=reach)
        for _, r in d.iterrows():
            ax.annotate(r.patch_id.replace(reach + "_", ""),
                        (r.dNDVI_median, r.dNBR_median),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axvline(0, linewidth=0.8, linestyle="--")
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Median dNDVI (POST − PRE)")
    ax.set_ylabel("Median dNBR (POST − PRE)")
    ax.set_title("Top-10 candidate patches per reach: NDVI–NBR change space")
    ax.legend(title="Reach")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"Figure2_Top20_dNDVI_vs_dNBR.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: strict QA coverage. One axes only.
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for reach, marker in [("R05", "o"), ("R06", "^")]:
        d = master.loc[master.zone_id == reach]
        sizes = 45 + 500 * np.sqrt(d.area_km2 / master.area_km2.max())
        ax.scatter(d.PRE_strict_QA_match_coverage_pct,
                   d.POST_strict_QA_match_coverage_pct,
                   s=sizes, marker=marker, alpha=0.78, label=reach)
        for _, r in d.iterrows():
            ax.annotate(r.patch_id.replace(reach + "_", ""),
                        (r.PRE_strict_QA_match_coverage_pct,
                         r.POST_strict_QA_match_coverage_pct),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.plot([0, 100], [0, 100], linewidth=0.8, linestyle="--")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("PRE strict-QA matched coverage (%)")
    ax.set_ylabel("POST strict-QA matched coverage (%)")
    ax.set_title("Top-20 source-reconciled patches: strict-QA support")
    ax.legend(title="Reach")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"Figure3_Top20_PRE_POST_QA_coverage.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: dNDVI ordered within each reach, still a single axes.
    ordered = master.sort_values(["zone_id", "area_rank_in_reach"]).copy()
    labels = ordered.patch_id.tolist()
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for reach, marker in [("R05", "o"), ("R06", "^")]:
        mask = ordered.zone_id.eq(reach).to_numpy()
        ax.scatter(x[mask], ordered.loc[mask, "dNDVI_median"], marker=marker,
                   s=65, label=reach)
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.axhline(-0.15, linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Median dNDVI (POST − PRE)")
    ax.set_xlabel("Patch (area-rank order within reach)")
    ax.set_title("Top-20 candidate patches: median NDVI change")
    ax.legend(title="Reach")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"Figure4_Top20_dNDVI_by_patch.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"Publication figures written to: {fig_dir}")
    return fig_dir


# -----------------------------------------------------------------------------
# CEMS Rapid Mapping independent product audit
# -----------------------------------------------------------------------------

def _activation_code(record):
    for key in ["activationCode", "activation_code", "code"]:
        value = record.get(key)
        if value:
            return str(value).strip().upper()
    aois = record.get("aois") or []
    for aoi in aois:
        value = aoi.get("activationCode") or aoi.get("activation_code")
        if value:
            return str(value).strip().upper()
    return None


def fetch_cems_activation(cfg, out):
    import requests

    cache = out / "CEMS_EMSR927_activation.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8")), "CACHED"

    session = requests.Session()
    base = cfg["cems_public_api"]
    offset = 0
    target = cfg["cems_activation_code"].upper()
    for page in range(cfg["cems_max_pages"]):
        response = session.get(
            base,
            params={"limit": cfg["cems_page_size"], "offset": offset},
            timeout=cfg["cems_request_timeout_seconds"],
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", payload if isinstance(payload, list) else [])
        require(isinstance(results, list), "Unexpected CEMS activation API response schema")
        for rec in results:
            if _activation_code(rec) == target:
                write_json(cache, rec)
                return rec, "DOWNLOADED"
        nxt = payload.get("next") if isinstance(payload, dict) else None
        if not nxt or not results:
            break
        offset += len(results)
    raise RuntimeError(f"CEMS activation {target} was not found in the public Rapid Mapping API")


def _parse_wkt(value):
    from shapely import wkt
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.upper().startswith("SRID=") and ";" in text:
            text = text.split(";", 1)[1]
        g = wkt.loads(text)
        return g if not g.is_empty else None
    except Exception:
        return None


def _layer_url(layer, base_url):
    for key in ["json", "geojson", "downloadPath", "download_path"]:
        value = layer.get(key)
        if isinstance(value, str) and value.strip():
            return urljoin(base_url, value.strip())
    return None


def download_json_limited(url, path, cfg):
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), "CACHED"
    with requests.get(url, stream=True, timeout=cfg["cems_request_timeout_seconds"]) as r:
        r.raise_for_status()
        declared = r.headers.get("Content-Length")
        if declared and int(declared) > cfg["cems_max_geojson_bytes"]:
            raise RuntimeError(f"CEMS layer exceeds size safety limit: {declared} bytes")
        chunks, total = [], 0
        for block in r.iter_content(1024 * 1024):
            if not block:
                continue
            total += len(block)
            if total > cfg["cems_max_geojson_bytes"]:
                raise RuntimeError("CEMS layer exceeded size safety limit during download")
            chunks.append(block)
    raw = b"".join(chunks)
    payload = json.loads(raw.decode("utf-8"))
    path.write_bytes(raw)
    return payload, "DOWNLOADED"


def cems_overlap_analysis(priority, geom_by_id, activation, cfg, out):
    import pandas as pd
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer

    base = cfg["cems_public_api"]
    code = cfg["cems_activation_code"]
    cems_dir = out / "CEMS_layers"
    cems_dir.mkdir(parents=True, exist_ok=True)
    to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True).transform

    patch_ids = priority.patch_id.tolist()
    aoi_rows, layer_rows, overlap_rows, errors = [], [], [], []

    aois = activation.get("aois") or []
    for aoi_idx, aoi in enumerate(aois, start=1):
        aoi_num = aoi.get("number", aoi_idx)
        aoi_name = aoi.get("name", f"AOI_{aoi_num}")
        aoi_geom = _parse_wkt(aoi.get("extent"))
        overlap_patches = []
        if aoi_geom is not None:
            overlap_patches = [pid for pid in patch_ids if aoi_geom.intersects(geom_by_id[pid])]
        aoi_rows.append({
            "activation": code,
            "aoi_number": aoi_num,
            "aoi_name": aoi_name,
            "extent_available": aoi_geom is not None,
            "priority_patch_intersections": len(overlap_patches),
            "priority_patch_ids": ";".join(overlap_patches),
        })
        if not overlap_patches:
            continue

        products = aoi.get("products") or []
        for product in products:
            ptype = str(product.get("type", ""))
            product_id = product.get("id")
            feasible = product.get("feasible")
            version = product.get("version") or {}
            status_code = version.get("statusCode") if isinstance(version, dict) else None
            layers = product.get("layers") or []
            for li, layer in enumerate(layers, start=1):
                lname = str(layer.get("name", f"layer_{li}"))
                fmt = str(layer.get("format", ""))
                url = _layer_url(layer, base)
                relevant_name = any(k.lower() in lname.lower() for k in cfg["cems_relevant_keywords"])
                potentially_relevant = relevant_name or ptype.upper() in {"GRA", "DEL"}
                layer_row = {
                    "activation": code, "aoi_number": aoi_num, "aoi_name": aoi_name,
                    "product_id": product_id, "product_type": ptype, "product_feasible": feasible,
                    "version_status": status_code, "layer_name": lname, "layer_format": fmt,
                    "layer_json_url": url, "keyword_relevant_name": relevant_name,
                    "potentially_crisis_relevant": potentially_relevant,
                    "download_status": "NOT_ATTEMPTED", "feature_count": None,
                }
                if not url or ("json" not in layer or not layer.get("json")):
                    layer_row["download_status"] = "NO_DIRECT_GEOJSON_URL"
                    layer_rows.append(layer_row)
                    continue
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"AOI{aoi_num}_{ptype}_{product_id}_{li}_{Path(lname).name}")
                if not safe_name.lower().endswith((".json", ".geojson")):
                    safe_name += ".geojson"
                target = cems_dir / safe_name
                try:
                    gj, status = download_json_limited(url, target, cfg)
                    feats = gj.get("features", []) if isinstance(gj, dict) else []
                    layer_row["download_status"] = status
                    layer_row["feature_count"] = len(feats)
                    layer_row["local_path"] = str(target)
                    for pid in overlap_patches:
                        patch = geom_by_id[pid]
                        patch_utm = shapely_transform(to_utm, patch)
                        feature_hits = 0
                        poly_area = 0.0
                        line_len = 0.0
                        point_hits = 0
                        attribute_examples = []
                        geom_types = set()
                        for feat in feats:
                            geometry = feat.get("geometry")
                            if not geometry:
                                continue
                            try:
                                g = shape(geometry)
                            except Exception:
                                continue
                            if g.is_empty or not g.intersects(patch):
                                continue
                            feature_hits += 1
                            geom_types.add(g.geom_type)
                            inter = g.intersection(patch)
                            if not inter.is_empty:
                                inter_utm = shapely_transform(to_utm, inter)
                                if inter.geom_type in ["Polygon", "MultiPolygon"]:
                                    poly_area += float(inter_utm.area)
                                elif "LineString" in inter.geom_type:
                                    line_len += float(inter_utm.length)
                                elif "Point" in inter.geom_type:
                                    point_hits += 1
                            if len(attribute_examples) < 3:
                                props = feat.get("properties", {}) or {}
                                attribute_examples.append({k: props[k] for k in list(props)[:12]})
                        if feature_hits:
                            overlap_rows.append({
                                "patch_id": pid, "activation": code,
                                "aoi_number": aoi_num, "aoi_name": aoi_name,
                                "product_id": product_id, "product_type": ptype,
                                "layer_name": lname, "keyword_relevant_name": relevant_name,
                                "potentially_crisis_relevant": potentially_relevant,
                                "intersecting_feature_count": feature_hits,
                                "geometry_types": ";".join(sorted(geom_types)),
                                "polygon_intersection_area_m2": poly_area,
                                "polygon_intersection_pct_patch": 100 * poly_area / patch_utm.area if patch_utm.area > 0 else None,
                                "line_intersection_length_m": line_len,
                                "point_intersections": point_hits,
                                "attribute_examples_json": json.dumps(attribute_examples, default=str),
                            })
                except Exception as exc:
                    layer_row["download_status"] = "FAILED"
                    layer_row["error"] = safe_error(exc)
                    errors.append({"scope": "CEMS_LAYER", "aoi_number": aoi_num, "product_id": product_id,
                                   "layer_name": lname, "url": url, "error": safe_error(exc)})
                layer_rows.append(layer_row)

    aoi_df = pd.DataFrame(aoi_rows)
    layer_df = pd.DataFrame(layer_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    error_df = pd.DataFrame(errors)
    aoi_df.to_csv(out / "CEMS_AOI_coverage.csv", index=False)
    layer_df.to_csv(out / "CEMS_layer_inventory.csv", index=False)
    overlap_df.to_csv(out / "CEMS_patch_vector_intersections.csv", index=False)
    error_df.to_csv(out / "CEMS_download_errors.csv", index=False)

    summary_rows = []
    for pid in patch_ids:
        covered = False
        names = []
        if len(aoi_df):
            hit = aoi_df[aoi_df["priority_patch_ids"].fillna("").str.split(";").apply(lambda xs: pid in xs)]
            covered = len(hit) > 0
            names = hit.aoi_name.astype(str).tolist()
        if len(overlap_df):
            op = overlap_df[overlap_df.patch_id == pid]
        else:
            op = pd.DataFrame()
        rel = op[op.potentially_crisis_relevant == True] if len(op) and "potentially_crisis_relevant" in op else pd.DataFrame()
        if not covered:
            status = "OUTSIDE_CEMS_MAPPED_AOI"
        elif len(op) == 0:
            status = "CEMS_AOI_COVERED_NO_DOWNLOADED_VECTOR_INTERSECTION"
        elif len(rel) == 0:
            status = "CEMS_VECTOR_INTERSECTION_ONLY_NONKEYWORD_LAYER"
        else:
            status = "CEMS_POTENTIALLY_RELEVANT_VECTOR_INTERSECTION"
        summary_rows.append({
            "patch_id": pid,
            "cems_aoi_covered": covered,
            "cems_aoi_names": ";".join(names),
            "cems_intersecting_layers": int(len(op)),
            "cems_keyword_relevant_intersecting_layers": int(len(rel)),
            "cems_intersecting_feature_count": int(op.intersecting_feature_count.sum()) if len(op) else 0,
            "cems_polygon_intersection_area_m2_sum_nonexclusive": float(op.polygon_intersection_area_m2.sum()) if len(op) else 0.0,
            "cems_status": status,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "CEMS_patch_summary.csv", index=False)
    return summary, overlap_df, error_df


# -----------------------------------------------------------------------------
# Landsat 8/9 independent-sensor temporal check
# -----------------------------------------------------------------------------

def initialize_ee(cfg):
    import ee
    try:
        ee.Initialize(project=cfg["ee_project"])
        ee.Number(1).getInfo()
    except Exception:
        print("Earth Engine needs authentication. Use the account authorized for the project.")
        ee.Authenticate()
        ee.Initialize(project=cfg["ee_project"])
        ee.Number(1).getInfo()
    print("Earth Engine initialized:", cfg["ee_project"])


def ee_inventory_plain(collection, maximum):
    import ee
    n = int(collection.size().getInfo())
    require(n <= maximum, f"Landsat scene count {n} exceeds safety limit {maximum}")
    if n == 0:
        return []
    props = [
        "SPACECRAFT_ID", "WRS_PATH", "WRS_ROW", "CLOUD_COVER",
        "PROCESSING_LEVEL", "LANDSAT_PRODUCT_ID", "DATE_ACQUIRED"
    ]
    def as_dict(el):
        im = ee.Image(el)
        return (im.toDictionary(props)
                .set("audit_scene_index", im.get("system:index"))
                .set("audit_time_start_ms", im.get("system:time_start"))
                .set("audit_asset_id", im.id()))
    rows = collection.toList(n).map(as_dict).getInfo()
    out = []
    for r in rows:
        idx = r.get("audit_scene_index")
        aid = r.get("audit_asset_id")
        require(idx and aid, "Landsat inventory row missing scene or asset identifier")
        if "/" not in aid:
            # We do not infer which collection from a bare ID here; the caller adds it.
            r["audit_element_id"] = aid
        out.append(r)
    return out


def landsat_collection_records(cfg, union_geom):
    import ee
    from shapely.geometry import mapping

    region = ee.Geometry(mapping(union_geom), CRS_WGS84, False)
    rows = []
    for collection_id in cfg["landsat_collections"]:
        col = (ee.ImageCollection(collection_id)
               .filterDate(cfg["landsat_start"], cfg["landsat_end_exclusive"])
               .filterBounds(region))
        got = ee_inventory_plain(col, cfg["landsat_max_scene_assets"])
        for r in got:
            idx = str(r["audit_scene_index"])
            aid = str(r["audit_asset_id"])
            if "/" not in aid:
                require(aid == idx, "Bare Landsat element ID conflicts with system:index")
                aid = collection_id.rstrip("/") + "/" + idx
            r["asset_id"] = aid
            r["collection_id"] = collection_id
            r["acquired_utc"] = iso(datetime.fromtimestamp(float(r["audit_time_start_ms"]) / 1000, timezone.utc))
            spacecraft = str(r.get("SPACECRAFT_ID") or "UNKNOWN")
            path = r.get("WRS_PATH")
            row = r.get("WRS_ROW")
            r["track_key"] = f"{spacecraft}|{path}|{row}"
            rows.append(r)
    rows.sort(key=lambda r: (r["acquired_utc"], r["asset_id"]))
    # Distinct assets only.
    require(len({r["asset_id"] for r in rows}) == len(rows), "Duplicate Landsat assets across queried collections")
    return rows


def landsat_processed_image(asset_id):
    import ee
    raw = ee.Image(asset_id)
    qa = raw.select("QA_PIXEL")
    # C2 QA_PIXEL bits 0..5: fill, dilated cloud, cirrus, cloud, cloud shadow, snow.
    qa_clear = qa.bitwiseAnd(0b111111).eq(0)
    sat_clear = raw.select("QA_RADSAT").eq(0)
    clear = qa_clear.And(sat_clear).rename("clear")
    sr = raw.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(0.0000275).add(-0.2)
    sr = sr.rename(["B2", "B3", "B4", "B5", "B6", "B7"])

    # Avoid ee.Image.normalizedDifference() here because that helper masks
    # pixels when either source band is negative. Collection-2 SR can contain
    # physically valid negative values after scaling/offset, especially over
    # dark surfaces. We retain them and mask only near-zero denominators.
    def ratio(a, b, name):
        aa, bb = sr.select(a), sr.select(b)
        den = aa.add(bb)
        return aa.subtract(bb).divide(den).updateMask(den.abs().gt(1e-6)).rename(name)

    ndvi = ratio("B5", "B4", "NDVI")
    mndwi = ratio("B3", "B6", "MNDWI")
    nbr = ratio("B5", "B7", "NBR")
    return ee.Image.cat([sr, ndvi, mndwi, nbr, clear]).updateMask(clear)


def reduce_landsat_scene_to_patch(processed, patch_ee, cfg):
    import ee
    scale = cfg["landsat_scale_m"]
    total = ee.Image.constant(1).rename("total").reduceRegion(
        reducer=ee.Reducer.count(), geometry=patch_ee, scale=scale,
        maxPixels=1_000_000, bestEffort=False
    ).get("total")
    total = int(ee.Number(total).getInfo() or 0)
    stats = processed.select(["B4", "B5", "B6", "B7", "NDVI", "MNDWI", "NBR"]).reduceRegion(
        reducer=ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=patch_ee, scale=scale, maxPixels=1_000_000, bestEffort=False
    ).getInfo()
    valid = int(stats.get("NDVI_count") or 0)
    row = {
        "total_pixel_centres": total,
        "valid_pixel_centres": valid,
        "clear_fraction": valid / total if total else 0.0,
    }
    for name in ["NDVI", "MNDWI", "NBR", "B4", "B5", "B6", "B7"]:
        value = stats.get(name + "_median")
        row[name + "_median"] = float(value) if value is not None else None
    return row


def landsat_pair_delta(pre_asset, post_asset, patch_ee, cfg):
    import ee
    pre = landsat_processed_image(pre_asset)
    post = landsat_processed_image(post_asset)
    common = pre.select("NDVI").mask().And(post.select("NDVI").mask())
    diff = ee.Image.cat([
        post.select("NDVI").subtract(pre.select("NDVI")).rename("dNDVI"),
        post.select("MNDWI").subtract(pre.select("MNDWI")).rename("dMNDWI"),
        post.select("NBR").subtract(pre.select("NBR")).rename("dNBR"),
        common.rename("common"),
    ]).updateMask(common)
    scale = cfg["landsat_scale_m"]
    total = int(ee.Number(ee.Image.constant(1).rename("total").reduceRegion(
        ee.Reducer.count(), patch_ee, scale, maxPixels=1_000_000, bestEffort=False
    ).get("total")).getInfo() or 0)
    vals = diff.reduceRegion(
        ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True),
        patch_ee, scale, maxPixels=1_000_000, bestEffort=False
    ).getInfo()
    common_n = int(vals.get("dNDVI_count") or 0)
    return {
        "common_clear_pixel_centres": common_n,
        "common_clear_fraction": common_n / total if total else 0.0,
        "landsat_dNDVI_median": float(vals["dNDVI_median"]) if vals.get("dNDVI_median") is not None else None,
        "landsat_dMNDWI_median": float(vals["dMNDWI_median"]) if vals.get("dMNDWI_median") is not None else None,
        "landsat_dNBR_median": float(vals["dNBR_median"]) if vals.get("dNBR_median") is not None else None,
    }


def landsat_analysis(priority, geom_by_id, cfg, out):
    import ee
    import pandas as pd
    from shapely.ops import unary_union
    from shapely.geometry import mapping

    initialize_ee(cfg)
    union_geom = unary_union([geom_by_id[p] for p in priority.patch_id])
    scenes = landsat_collection_records(cfg, union_geom)
    write_json(out / "Landsat_scene_inventory.json", scenes)
    pd.DataFrame(scenes).to_csv(out / "Landsat_scene_inventory.csv", index=False)
    print(f"Landsat inventory: {len(scenes)} scene assets across Landsat 8/9.")

    per_scene_rows = []
    failures = []
    processed_cache = {}
    for _, prow in priority.iterrows():
        pid = prow.patch_id
        patch_ee = ee.Geometry(mapping(geom_by_id[pid]), CRS_WGS84, False)
        print(f"  Landsat patch: {pid}")
        for scene in scenes:
            try:
                asset = scene["asset_id"]
                if asset not in processed_cache:
                    processed_cache[asset] = landsat_processed_image(asset)
                stats = reduce_landsat_scene_to_patch(processed_cache[asset], patch_ee, cfg)
                per_scene_rows.append({
                    "patch_id": pid,
                    "asset_id": asset,
                    "acquired_utc": scene["acquired_utc"],
                    "SPACECRAFT_ID": scene.get("SPACECRAFT_ID"),
                    "WRS_PATH": scene.get("WRS_PATH"),
                    "WRS_ROW": scene.get("WRS_ROW"),
                    "track_key": scene["track_key"],
                    "CLOUD_COVER_scene_pct": scene.get("CLOUD_COVER"),
                    **stats,
                })
            except Exception as exc:
                failures.append({"patch_id": pid, "asset_id": scene.get("asset_id"), "error": safe_error(exc)})

    scene_df = pd.DataFrame(per_scene_rows)
    fail_df = pd.DataFrame(failures)
    scene_df.to_csv(out / "Landsat_patch_scene_metrics.csv", index=False)
    fail_df.to_csv(out / "Landsat_failures.csv", index=False)

    pair_rows = []
    event = utc(EVENT_TIME_UTC)
    for _, prow in priority.iterrows():
        pid = prow.patch_id
        d = scene_df[scene_df.patch_id == pid].copy()
        if d.empty:
            pair_rows.append({"patch_id": pid, "landsat_pair_status": "NO_LANDSAT_METRICS"})
            continue
        d["when"] = d.acquired_utc.map(utc)
        d["quality_ok"] = ((d.clear_fraction >= cfg["landsat_min_clear_fraction"]) &
                           (d.valid_pixel_centres >= cfg["landsat_min_valid_pixels"]))
        candidates = []
        for track, g in d.groupby("track_key"):
            q = g[g.quality_ok]
            pre = q[q.when < event].sort_values("when")
            post = q[q.when >= event].sort_values("when")
            if pre.empty or post.empty:
                continue
            a = pre.iloc[-1]
            b = post.iloc[0]
            candidates.append((
                -min(float(a.clear_fraction), float(b.clear_fraction)),
                (event - a.when).total_seconds() + (b.when - event).total_seconds(),
                -min(int(a.valid_pixel_centres), int(b.valid_pixel_centres)),
                str(track), a, b
            ))
        if not candidates:
            pair_rows.append({
                "patch_id": pid,
                "landsat_pair_status": "NO_SAME_PLATFORM_TRACK_PRE_POST_PAIR_AT_CONFIGURED_QUALITY",
                "landsat_scene_count": len(d),
                "landsat_quality_scene_count": int(d.quality_ok.sum()),
            })
            continue
        _, _, _, track, a, b = sorted(candidates, key=lambda x: x[:4])[0]
        patch_ee = ee.Geometry(mapping(geom_by_id[pid]), CRS_WGS84, False)
        delta = landsat_pair_delta(a.asset_id, b.asset_id, patch_ee, cfg)
        common_ok = (delta["common_clear_fraction"] >= cfg["landsat_common_clear_min_fraction"] and
                     delta["common_clear_pixel_centres"] >= cfg["landsat_min_valid_pixels"])
        row = {
            "patch_id": pid,
            "landsat_pair_status": "PAIR_AVAILABLE" if common_ok else "PAIR_AVAILABLE_LOW_COMMON_CLEAR_SUPPORT",
            "landsat_track_key": track,
            "landsat_pre_asset": a.asset_id,
            "landsat_post_asset": b.asset_id,
            "landsat_pre_utc": a.acquired_utc,
            "landsat_post_utc": b.acquired_utc,
            "landsat_pre_clear_fraction": float(a.clear_fraction),
            "landsat_post_clear_fraction": float(b.clear_fraction),
            "landsat_pre_valid_pixel_centres": int(a.valid_pixel_centres),
            "landsat_post_valid_pixel_centres": int(b.valid_pixel_centres),
            "landsat_pre_NDVI_median": a.NDVI_median,
            "landsat_post_NDVI_median": b.NDVI_median,
            **delta,
        }
        # Compare only signs, not magnitudes, because Sentinel-2 and Landsat have
        # different spectral response functions and spatial grids.
        s2_dndvi = float(prow.dNDVI_median)
        s2_dmndwi = float(prow.dMNDWI_median)
        s2_dnbr = float(prow.dNBR_median)
        def sign_agree(x, y):
            if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)):
                return None
            return bool(math.copysign(1, float(x)) == math.copysign(1, float(y))) if float(x) != 0 and float(y) != 0 else bool(float(x) == float(y))
        row["dNDVI_sign_agrees_with_Sentinel2"] = sign_agree(row["landsat_dNDVI_median"], s2_dndvi)
        row["dMNDWI_sign_agrees_with_Sentinel2"] = sign_agree(row["landsat_dMNDWI_median"], s2_dmndwi)
        row["dNBR_sign_agrees_with_Sentinel2"] = sign_agree(row["landsat_dNBR_median"], s2_dnbr)
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out / "Landsat_same_track_pair_summary.csv", index=False)
    return pair_df, scene_df, fail_df


# -----------------------------------------------------------------------------
# Merge / report / archive
# -----------------------------------------------------------------------------

def merge_external_master(priority, cems_summary, landsat_pairs, out):
    import pandas as pd

    merged = priority.copy()
    merged = merged.merge(cems_summary, on="patch_id", how="left", validate="one_to_one")
    merged = merged.merge(landsat_pairs, on="patch_id", how="left", validate="one_to_one")

    def data_status(row):
        parts = []
        c = str(row.get("cems_status", ""))
        if c:
            parts.append(c)
        l = str(row.get("landsat_pair_status", ""))
        if l:
            parts.append(l)
        return ";".join(parts) if parts else "NO_EXTERNAL_DATA_STATUS"

    merged["external_data_status"] = merged.apply(data_status, axis=1)
    merged["process_interpretation"] = "NOT_ASSIGNED_BY_EXTERNAL_AUDIT"
    merged["damage_validation"] = "NOT_ASSIGNED_BY_EXTERNAL_AUDIT"
    merged.to_csv(out / "PRIORITY5_EXTERNAL_CORROBORATION_MASTER.csv", index=False)
    with pd.ExcelWriter(out / "PRIORITY5_EXTERNAL_CORROBORATION_MASTER.xlsx", engine="openpyxl") as xw:
        merged.to_excel(xw, sheet_name="External_Master", index=False)
        cems_summary.to_excel(xw, sheet_name="CEMS_Summary", index=False)
        landsat_pairs.to_excel(xw, sheet_name="Landsat_Pairs", index=False)

    print("\nEXTERNAL CORROBORATION MASTER — DESCRIPTIVE, NOT DAMAGE VALIDATION")
    cols = [c for c in [
        "patch_id", "zone_id", "area_km2", "dNDVI_median",
        "cems_aoi_covered", "cems_status",
        "landsat_pair_status", "landsat_dNDVI_median",
        "dNDVI_sign_agrees_with_Sentinel2", "external_data_status"
    ] if c in merged.columns]
    print(merged[cols].to_string(index=False))
    return merged


def write_readme(out, cfg, priority_count):
    text = f"""LHENDE 2026 PRIORITY-PATCH EXTERNAL CORROBORATION v{VERSION}

Input: completed Top-20 patch review. Priority patches are those with
batch_evidence_status={cfg['priority_status']}. This is a technical evidence
condition, not a damage ranking. Priority patch count: {priority_count}.

CEMS: public Rapid Mapping activation {cfg['cems_activation_code']} is queried.
Patch/AOI and patch/vector intersections are descriptive. Inspect product type,
layer name, attributes, and product metadata before assigning a process or
damage label. Absence of an intersecting CEMS feature is not evidence of no
impact when mapping coverage or layer availability is incomplete.

Landsat: Collection-2 Level-2 Landsat 8/9 is independent of the Sentinel-2
optical composite. Pair selection uses quality/time only and requires the same
spacecraft + WRS path/row. Spectral sign agreement is descriptive and does not
establish event causality. Small patches may have few 30-m pixels.

Figures: descriptive plots use the frozen Top-20 table. Marker size reflects
patch area only. The Top-20 set is size-targeted and is not a representative
accuracy sample.

No original detections, geometries, thresholds, validation_status fields, or
process labels are modified.
"""
    (out / "README_EXTERNAL_CORROBORATION.txt").write_text(text, encoding="utf-8")

def checksums_and_zip(out):
    import pandas as pd
    files = [p for p in sorted(out.rglob("*")) if p.is_file()
             and p.name not in {"output_checksums.csv", "Lhende_External_Corroboration_Bundle.zip"}]
    pd.DataFrame([
        {"file": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in files
    ]).to_csv(out / "output_checksums.csv", index=False)
    zip_path = out / "Lhende_External_Corroboration_Bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, arcname=str(p.relative_to(out)))
    return zip_path


def run_external_corroboration(config=None, source_code_path=None):
    dependencies()
    import pandas as pd
    cfg = dict(CONFIG if config is None else config)
    mount_drive_if_needed()
    out = resolve_output(cfg)
    write_json(out / "run_configuration.json", {"version": VERSION, "config": cfg,
                                                  "created_utc": iso(datetime.now(timezone.utc))})
    if source_code_path and Path(str(source_code_path)).is_file():
        shutil.copy2(source_code_path, out / Path(str(source_code_path)).name)

    master, priority, geom_by_id, props_by_id = load_frozen_top20(cfg, out)
    make_figures(master, out, cfg)

    print("\nStage 1/2 — Copernicus EMSR927 independent product audit...")
    try:
        activation, activation_source = fetch_cems_activation(cfg, out)
        cems_summary, cems_overlap, cems_errors = cems_overlap_analysis(priority, geom_by_id, activation, cfg, out)
        print(f"CEMS activation loaded ({activation_source}); priority AOI/vector assessment complete.")
    except Exception as exc:
        print("CEMS audit unavailable; retained explicitly:", safe_error(exc))
        pd.DataFrame([{"scope": "CEMS_ACTIVATION", "error": safe_error(exc)}]).to_csv(out / "CEMS_download_errors.csv", index=False)
        cems_summary = pd.DataFrame([{
            "patch_id": pid, "cems_aoi_covered": None, "cems_aoi_names": "",
            "cems_intersecting_layers": 0, "cems_keyword_relevant_intersecting_layers": 0,
            "cems_intersecting_feature_count": 0,
            "cems_polygon_intersection_area_m2_sum_nonexclusive": 0.0,
            "cems_status": "CEMS_AUDIT_UNAVAILABLE"
        } for pid in priority.patch_id])
        cems_summary.to_csv(out / "CEMS_patch_summary.csv", index=False)

    print("\nStage 2/2 — Landsat 8/9 independent-sensor check...")
    try:
        landsat_pairs, landsat_scene_metrics, landsat_failures = landsat_analysis(priority, geom_by_id, cfg, out)
    except Exception as exc:
        print("Landsat audit unavailable; retained explicitly:", safe_error(exc))
        pd.DataFrame([{"scope": "LANDSAT_AUDIT", "error": safe_error(exc)}]).to_csv(out / "Landsat_failures.csv", index=False)
        landsat_pairs = pd.DataFrame([{
            "patch_id": pid,
            "landsat_pair_status": "LANDSAT_AUDIT_UNAVAILABLE",
            "landsat_dNDVI_median": None,
            "landsat_dMNDWI_median": None,
            "landsat_dNBR_median": None,
            "dNDVI_sign_agrees_with_Sentinel2": None,
            "dMNDWI_sign_agrees_with_Sentinel2": None,
            "dNBR_sign_agrees_with_Sentinel2": None,
        } for pid in priority.patch_id])
        landsat_pairs.to_csv(out / "Landsat_same_track_pair_summary.csv", index=False)

    merged = merge_external_master(priority, cems_summary, landsat_pairs, out)
    write_readme(out, cfg, len(priority))
    zip_path = checksums_and_zip(out)

    print("\nCOMPLETE")
    print("Saved:", out)
    print("Bundle:", zip_path)
    print("No original detections, geometries, thresholds, validation labels, or process labels were changed.")
    print("External results are descriptive corroboration/coverage evidence, not automatic damage validation.")
    return {
        "output_dir": str(out),
        "priority_patches": priority,
        "external_master": merged,
        "bundle": str(zip_path),
    }


if __name__ == "__main__":
    _source = globals().get("EXTERNAL_CORROBORATION_SCRIPT_PATH") or globals().get("__file__")
    EXTERNAL_CORROBORATION_RESULTS = run_external_corroboration(CONFIG, source_code_path=_source)
