from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer


def _same_grid(a, b):
    return (a.crs == b.crs and a.width == b.width and a.height == b.height and a.transform.almost_equals(b.transform))


def _read_rms(ds):
    desc = [(x or "").strip().lower() for x in ds.descriptions]
    for i, name in enumerate(desc, start=1):
        if name in {"spectral_rms", "rms", "spectral rms"}:
            return ds.read(i).astype("float64"), i
    if ds.count == 1:
        return ds.read(1).astype("float64"), 1
    if ds.count >= 13:
        return ds.read(13).astype("float64"), 13
    raise ValueError("Cannot identify spectral RMS band. Name it 'spectral_rms' or supply a one-band raster.")


def _valid(data, nodata):
    ok = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        ok &= data != nodata
    return ok


def quantify_concurrence(sar_path, rms_path, zones_geojson, output_dir, threshold=0.08,
                         sensitivity=(0.06, 0.08, 0.10), zone_field="zone_id"):
    """Generic, no-resampling SAR-Class3 + optical-RMS concurrence.

    SAR codes: 0 none, 1 control-only, 2 event+control, 3 event-only.
    Missing optical remains unknown; it is never converted to no-change.
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(sar_path) as sar, rasterio.open(rms_path) as rms_ds:
        if not _same_grid(sar, rms_ds):
            raise ValueError("SAR and RMS rasters are not on the exact same grid; resampling is intentionally refused.")
        sar_arr = sar.read(1).astype("int32")
        rms_arr, rms_band = _read_rms(rms_ds)
        sar_valid = _valid(sar_arr.astype("float64"), sar.nodata) & np.isin(sar_arr, [0,1,2,3])
        rms_valid = _valid(rms_arr, rms_ds.nodata)
        joint = sar_valid & rms_valid
        profile = sar.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        m01 = np.full(sar_arr.shape, 255, np.uint8)
        m01[joint] = 0
        m01[joint & (sar_arr == 3) & (rms_arr >= threshold)] = 1
        with rasterio.open(output_dir/"M01_concurrence.tif", "w", **profile) as dst:
            dst.write(m01, 1); dst.set_band_description(1, "SAR_Class3_AND_RMS_threshold")
        support = np.full(sar_arr.shape, 255, np.uint8)
        support[sar_valid & ~rms_valid] = 0
        support[joint] = 1
        with rasterio.open(output_dir/"M02_joint_optical_support.tif", "w", **profile) as dst:
            dst.write(support, 1); dst.set_band_description(1, "joint_observation_support")

        gj = json.loads(Path(zones_geojson).read_text(encoding="utf-8"))
        feats = gj.get("features", [])
        if not feats:
            raise ValueError("Zones GeoJSON has no features")
        zone_crs = CRS.from_user_input((gj.get("crs",{}).get("properties",{}).get("name") or "EPSG:4326"))
        raster_crs = CRS.from_user_input(sar.crs)
        transformer = Transformer.from_crs(zone_crs, raster_crs, always_xy=True)
        shapes = []
        zone_names = []
        for i, f in enumerate(feats, 1):
            zid = str(f.get("properties",{}).get(zone_field, f"zone_{i}"))
            geom = shape(f["geometry"])
            if zone_crs != raster_crs:
                geom = shp_transform(transformer.transform, geom)
            zone_names.append((i, zid))
            shapes.append((geom, i))
        zones = rasterize(shapes, out_shape=sar_arr.shape, transform=sar.transform, fill=0, all_touched=False, dtype="int32")
        pixel_km2 = abs(sar.transform.a * sar.transform.e) / 1e6
        rows=[]
        thresholds=sorted(set(float(x) for x in sensitivity) | {float(threshold)})
        for code,zid in zone_names:
            z = zones == code
            class3 = z & sar_valid & (sar_arr == 3)
            assessable = class3 & rms_valid
            base={
                "zone_id":zid,
                "sar_class3_total_km2":float(class3.sum()*pixel_km2),
                "sar_class3_optically_assessable_km2":float(assessable.sum()*pixel_km2),
                "sar_class3_optical_unavailable_km2":float((class3 & ~rms_valid).sum()*pixel_km2),
            }
            for t in thresholds:
                hit = assessable & (rms_arr >= t)
                rows.append({**base,"rms_threshold":t,"concurrence_km2":float(hit.sum()*pixel_km2),
                    "concurrence_pct_of_assessable_class3":float(100*hit.sum()/assessable.sum()) if assessable.sum() else np.nan,
                    "concurrence_pct_of_all_class3":float(100*hit.sum()/class3.sum()) if class3.sum() else np.nan})
        table=pd.DataFrame(rows)
        table.to_csv(output_dir/"concurrence_by_zone_and_threshold.csv", index=False)
        metadata={"sar":str(sar_path),"rms":str(rms_path),"zones":str(zones_geojson),"rms_band":rms_band,
                  "primary_threshold":threshold,"sensitivity":thresholds,"zone_field":zone_field,
                  "grid":{"crs":str(sar.crs),"width":sar.width,"height":sar.height,"transform":tuple(sar.transform)},
                  "interpretation":"Cross-sensor change candidates; not damage or causal proof."}
        (output_dir/"concurrence_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        return table
