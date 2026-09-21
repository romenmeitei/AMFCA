
# %% [markdown] cell 0
# # 26 August 2026 Lhende Khola–Bhote Koshi–Trishuli event
# ## Reproducible multi-sensor change analysis in Google Earth Engine
#
# **Version 1.4.1 ASSET-SAFE BATCH-MATERIALIZED — Google Colab / Earth Engine / QGIS**  
# **Temporal separator:** 26 August 2026, 02:52:10 UTC (08:37:10 Nepal Time)
#
# This notebook implements a complete workflow for:
#
# - identifying candidate source-zone disturbance;
# - building a closest-valid-pixel pre-event baseline and immediate post-event composite;
# - mapping snow/ice, surface water, vegetation, exposed rock, debris/mudflow, disturbed terrain, and built-up exposure;
# - assessing channel widening, temporary-water candidates, and downstream propagation over approximately 100 km;
# - integrating Sentinel-2, Sentinel-1, Landsat 9, Copernicus DEM, MERIT Hydro, JRC Global Surface Water, and ESA WorldCover;
# - batch-materialising the expensive categorical/mask graph as versioned Earth Engine assets before statistics, with a missing-asset-safe project-root check;
# - exporting rasters, vectors, area tables, transition tables, scene inventories, and a run manifest for QGIS;
# - optionally training a spatially separated Random Forest classifier using independent reference polygons.
#
# > **Interpretation rule:** the precise initiating mechanism and boundary are still provisional. Rule-based outputs are candidate impact layers, not field-validated ground truth. Missing/cloud-shadowed pixels are explicitly represented rather than interpreted as unchanged terrain.

# %% cell 1
# COLAB MAGIC: pip install -q -U earthengine-api geemap

# %% cell 2
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

print('Python:', sys.version.split()[0])
print('earthengine-api:', getattr(ee, '__version__', 'unknown'))
print('geemap:', getattr(geemap, '__version__', 'unknown'))
print('pandas:', pd.__version__)
print('numpy:', np.__version__)

# %% [markdown] cell 3
# ## 1. Authenticate and initialise Earth Engine
#
# The Cloud project below is inherited from the supplied notebook. Change it only when using another Earth Engine-enabled project.

# %% cell 4
EE_PROJECT = 'woven-name-441217-g5'

try:
    ee.Initialize(project=EE_PROJECT)
    print(f'Earth Engine already initialised: {EE_PROJECT}')
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
    print(f'Earth Engine authenticated and initialised: {EE_PROJECT}')

# %% [markdown] cell 5
# ## 2. Frozen configuration
#
# The notebook queries every intersecting Sentinel-2 tile in the study corridor. Verified source-tile IDs from the exploratory notebook are merged as an audit safeguard, but they are **not** the only scenes used. This avoids leaving the lower Trishuli corridor blank when it crosses other Sentinel-2 tiles.

# %% cell 6
# Event and time windows
EVENT_TIME_UTC = '2026-08-26T02:52:10Z'
EVENT_LABEL = '2026-08-26_Lhende_BhoteKoshi_event'

S2_PRIMARY_PRE_START = '2026-08-22'
S2_FALLBACK_PRE_START = '2026-08-10'
S2_FALLBACK_PRE_END = '2026-08-16'
S2_PRE_END = EVENT_TIME_UTC
S2_POST_START = EVENT_TIME_UTC
S2_POST_END = '2026-09-02'  # includes 1 September; end is exclusive

S1_PRE_START = '2026-07-20'
S1_PRE_END = EVENT_TIME_UTC
S1_POST_START = EVENT_TIME_UTC
S1_POST_END = '2026-09-02'

LANDSAT_PRE_START = '2026-08-01'
LANDSAT_PRE_END = EVENT_TIME_UTC
LANDSAT_POST_START = EVENT_TIME_UTC
LANDSAT_POST_END = '2026-09-02'

ADD_VERIFIED_S2_SCENES = True
S2_PRE_IDS = [
    '20260824T044659_20260824T045006_T45RUM',
    '20260824T050231_20260824T050638_T45RUM',
    '20260812T045701_20260812T050239_T45RUM',
]
S2_POST_IDS = [
    '20260827T045659_20260827T051017_T45RUM',
    '20260829T044701_20260829T045155_T45RUM',
    '20260831T045231_20260831T045835_T45RUM',
]

# Geometry
SOURCE_SEARCH_LON_LAT = (85.5150, 28.2930)  # exploratory centre, not a final source centroid
USGS_SIGNAL_LON_LAT = (85.5150, 28.2710)    # preliminary seismic catalogue reference
SOURCE_SEARCH_RADIUS_M = 12_000
CORRIDOR_HALF_WIDTH_M = 3_500

# Processing
ANALYSIS_SCALE_M = 20
DEM_SCALE_M = 30
HYDRO_SCALE_M = 90
TILE_SCALE = 4
MAX_PIXELS = 1e13
OUTPUT_CRS = 'EPSG:32645'

# Sentinel-2 quality
S2_SCENE_CLOUD_MAX = 98
S2_CLOUD_PROBABILITY_MAX = 55
MIN_COSINE_ILLUMINATION = 0.15

# Class/change thresholds: publication outputs must include sensitivity analysis.
NDSI_SNOW_THRESHOLD = 0.40
SNOW_MIN_NIR = 0.11
MNDWI_WATER_THRESHOLD = 0.15
WATER_MAX_NIR = 0.16
NDVI_VEGETATION_THRESHOLD = 0.35
NDVI_BARE_MAX = 0.25
BSI_BARE_MIN = -0.05
SPECTRAL_RMS_THRESHOLD = 0.08
SPECTRAL_ANGLE_THRESHOLD_RAD = 0.12
DNBR_ABS_THRESHOLD = 0.14
DNDVI_LOSS_THRESHOLD = -0.15
DNDSI_LOSS_THRESHOLD = -0.18
SAR_CHANGE_DB_THRESHOLD = 2.5
SAR_WATER_VV_DB = -17.0
SAR_WATER_VH_DB = -24.0
SAR_WATER_DROP_DB = -1.5
MAX_WATER_SLOPE_DEG = 25
MIN_CONNECTED_PIXELS = 8
MIN_VECTOR_AREA_M2 = 2_000
MIN_SOURCE_VECTOR_AREA_M2 = 5_000
MERIT_UPA_STREAM_KM2 = 5

# Optional user assets
CUSTOM_STUDY_AOI_ASSET = ''
CUSTOM_TRAINING_ASSET = ''
CEMS_REFERENCE_ASSET = ''
RUN_SUPERVISED_RF = bool(CUSTOM_TRAINING_ASSET)
RUN_OPTIONAL_HIGH_RES_INVENTORY = False
START_DRIVE_EXPORTS = False
MOUNT_GOOGLE_DRIVE = False
DOWNLOAD_LOCAL_AUDIT_ZIP = False

DRIVE_EXPORT_FOLDER = 'Lhende_2026_GEE_Exports'
LOCAL_OUTPUT_DIR = Path('/content/Lhende_2026_outputs')
RANDOM_SEED = 26082026
RF_SPLIT_PROPERTY = 'split'
RF_TRAIN_VALUE = 'train'
RF_VALIDATION_VALUE = 'validation'

CLASS_NAMES = {
    0: 'Unclassified / insufficient evidence',
    1: 'Snow / ice',
    2: 'Water',
    3: 'Vegetation',
    4: 'Exposed rock / stable bare terrain',
    5: 'Debris / mudflow candidate',
    6: 'Disturbed terrain candidate',
    7: 'Built-up / infrastructure context',
}
CLASS_PALETTE = ['7f7f7f', 'e8f4ff', '1f78b4', '33a02c', 'b2a06f', '8c510a', 'ff7f00', 'e31a1c']

print('Configuration loaded.')

# %% [markdown] cell 7
# ## 3. Study geometry: source search zone and approximately 100 km downstream corridor
#
# The waypoints are reproducible centreline anchors, not surveyed river-bank boundaries. The impact masks themselves are constrained with MERIT Hydro and historical water context. For the final manuscript, replace the default geometry with a field-checked or authoritative corridor asset through `CUSTOM_STUDY_AOI_ASSET`.

# %% cell 8
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))

# Approximate anchors following the Lhende/Bhote Koshi/Trishuli valley.
waypoints = [
    ('Source search centre', 85.51500, 28.29300),
    ('Rasuwagadhi / Timure', 85.37800, 28.27600),
    ('Syabrubesi', 85.33500, 28.16400),
    ('Betrawati', 85.19500, 27.97300),
    ('Trishuli / Bidur', 85.14370, 27.91930),
    ('Galchhi', 85.00046, 27.79740),
    ('Benighat', 84.79512, 27.78347),
]

source_search_point = ee.Geometry.Point(SOURCE_SEARCH_LON_LAT)
usgs_signal_point = ee.Geometry.Point(USGS_SIGNAL_LON_LAT)
source_search_aoi = source_search_point.buffer(SOURCE_SEARCH_RADIUS_M)

waypoint_features = []
cumulative_km = 0.0
for i, (name, lon, lat) in enumerate(waypoints):
    if i > 0:
        cumulative_km += haversine_km((waypoints[i-1][1], waypoints[i-1][2]), (lon, lat))
    waypoint_features.append(ee.Feature(ee.Geometry.Point([lon, lat]), {
        'waypoint_id': f'P{i:02d}', 'waypoint_name': name, 'km_from_source': round(cumulative_km, 3)
    }))
waypoint_fc = ee.FeatureCollection(waypoint_features)
reference_point_fc = ee.FeatureCollection([
    ee.Feature(usgs_signal_point, {'reference': 'USGS preliminary seismic signal location'}),
    ee.Feature(source_search_point, {'reference': 'Exploratory source search centre'}),
])

segment_geometries = []
reach_features = []
occupied = None
cumulative_km = 0.0
for i in range(len(waypoints) - 1):
    name_a, lon_a, lat_a = waypoints[i]
    name_b, lon_b, lat_b = waypoints[i + 1]
    segment_km = haversine_km((lon_a, lat_a), (lon_b, lat_b))
    raw_geom = ee.Geometry.LineString([[lon_a, lat_a], [lon_b, lat_b]]).buffer(CORRIDOR_HALF_WIDTH_M)
    if i == 0:
        raw_geom = raw_geom.union(source_search_aoi, 10)
    segment_geometries.append(raw_geom)
    zone_geom = raw_geom if occupied is None else raw_geom.difference(occupied, 10)
    reach_features.append(ee.Feature(zone_geom, {
        'zone_id': f'R{i+1:02d}',
        'zone_name': f'{name_a} to {name_b}',
        'start_km': round(cumulative_km, 3),
        'end_km': round(cumulative_km + segment_km, 3),
        'mid_km': round(cumulative_km + segment_km / 2, 3),
    }))
    occupied = raw_geom if occupied is None else occupied.union(raw_geom, 10)
    cumulative_km += segment_km

corridor_aoi = segment_geometries[0]
for geom in segment_geometries[1:]:
    corridor_aoi = corridor_aoi.union(geom, 10)
study_aoi = corridor_aoi.union(source_search_aoi, 10)

if CUSTOM_STUDY_AOI_ASSET:
    study_aoi = ee.FeatureCollection(CUSTOM_STUDY_AOI_ASSET).geometry()

reaches = ee.FeatureCollection(reach_features).filterBounds(study_aoi)
summary_zones = ee.FeatureCollection([
    ee.Feature(source_search_aoi.intersection(study_aoi, 10), {
        'zone_id': 'Z_SOURCE', 'zone_name': 'Source search zone', 'mid_km': 0.0
    }),
    ee.Feature(study_aoi, {
        'zone_id': 'Z_FULL', 'zone_name': 'Full study corridor', 'mid_km': cumulative_km / 2
    }),
])
all_zones = summary_zones.merge(reaches)

print(f'Approximate centreline length: {cumulative_km:.1f} km')
print('Reach count:', reaches.size().getInfo())

# %% cell 9
Map_context = geemap.Map(basemap='HYBRID')
Map_context.centerObject(study_aoi, 8)
Map_context.addLayer(study_aoi, {'color': 'red'}, 'Study corridor')
Map_context.addLayer(source_search_aoi, {'color': 'yellow'}, 'Source search zone')
Map_context.addLayer(reaches.style(color='00ffff', fillColor='00000000', width=2), {}, 'Non-overlapping reaches')
Map_context.addLayer(waypoint_fc.style(color='ffffff', pointSize=5), {}, 'Corridor waypoints')
Map_context.addLayer(reference_point_fc.style(color='ff00ff', pointSize=7), {}, 'Reference points')
Map_context.addLayerControl()
Map_context

# %% [markdown] cell 10
# ## 4. Utility functions

# %% cell 11
def safe_get_info(obj: Any, default: Any = None, label: str = 'Earth Engine object') -> Any:
    try:
        return obj.getInfo() if hasattr(obj, 'getInfo') else obj
    except Exception as exc:
        print(f'WARNING: unable to evaluate {label}: {exc}')
        return default


def prefix_bands(image: ee.Image, prefix: str) -> ee.Image:
    old_names = image.bandNames()
    new_names = old_names.map(lambda name: ee.String(prefix).cat(ee.String(name)))
    return image.rename(new_names)


def image_date_string(image: ee.Image) -> str:
    return safe_get_info(ee.Date(image.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss'), 'unknown')


def closest_collection(collection: ee.ImageCollection, start: str, end: str, target: str, limit: int = 3) -> ee.ImageCollection:
    target_ms = ee.Date(target).millis()
    def add_distance(img):
        img = ee.Image(img)
        distance = ee.Number(img.get('system:time_start')).subtract(target_ms).abs()
        return img.set('time_distance_ms', distance)
    return collection.filterDate(start, end).map(add_distance).sort('time_distance_ms').limit(limit)


def closest_valid_pixel_mosaic(collection: ee.ImageCollection, target: str, label: str) -> ee.Image:
    target_ms = ee.Date(target).millis()
    def add_distance(img):
        img = ee.Image(img)
        distance = ee.Number(img.get('system:time_start')).subtract(target_ms).abs()
        return img.set('time_distance_ms', distance)
    # mosaic() gives the last image highest priority: farthest first, closest last.
    ranked = collection.map(add_distance).sort('time_distance_ms', False)
    mosaic = ee.Image(ranked.mosaic())
    return mosaic.set({
        'composite_label': label,
        'target_time': target,
        'source_count': ranked.size(),
        'source_ids': ranked.aggregate_array('system:index'),
    })


def connected_mask(
    mask: ee.Image,
    minimum_pixels: int = MIN_CONNECTED_PIXELS,
    max_size: Optional[int] = None,
) -> ee.Image:
    """
    Retain foreground components containing at least ``minimum_pixels`` pixels.

    ``connectedPixelCount`` only needs to count up to the decision threshold.
    Counts larger than that threshold are not used anywhere in this study, so
    the previous 128-pixel neighbourhood rebuilt far more context than needed.
    Masking the zero background before counting is also essential.
    """
    minimum_pixels = max(1, int(minimum_pixels))
    if max_size is None:
        max_size = minimum_pixels
    max_size = max(minimum_pixels, int(max_size))

    foreground = (
        ee.Image(mask)
        .unmask(0)
        .neq(0)
        .toByte()
        .rename('foreground')
        .selfMask()
    )

    component_size = foreground.connectedPixelCount(
        maxSize=max_size,
        eightConnected=True,
    )

    return (
        foreground
        .updateMask(component_size.gte(minimum_pixels))
        .toByte()
    )


def safe_normalized_difference(image: ee.Image, a: str, b: str, name: str) -> ee.Image:
    numerator = image.select(a).subtract(image.select(b))
    denominator = image.select(a).add(image.select(b))
    return numerator.divide(denominator.where(denominator.abs().lt(1e-6), 1e-6)).rename(name)


def valid_fraction_ee(image: ee.Image, region: ee.Geometry, band: str = 'B4', scale: int = ANALYSIS_SCALE_M) -> ee.Number:
    pixel_area = ee.Image.pixelArea().rename('area')
    total = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=scale,
        bestEffort=True, maxPixels=MAX_PIXELS, tileScale=TILE_SCALE,
    ).get('area')
    valid = pixel_area.updateMask(image.select(band).mask()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=scale,
        bestEffort=True, maxPixels=MAX_PIXELS, tileScale=TILE_SCALE,
    ).get('area')
    total_number = ee.Number(ee.Algorithms.If(total, total, 0))
    valid_number = ee.Number(ee.Algorithms.If(valid, valid, 0))
    return ee.Number(ee.Algorithms.If(total_number.gt(0), valid_number.divide(total_number), 0))


def local_valid_fraction(image: ee.Image, region: ee.Geometry, band: str = 'B4', scale: int = ANALYSIS_SCALE_M) -> float:
    return float(safe_get_info(valid_fraction_ee(image, region, band, scale), 0.0, 'valid fraction'))

print('Utility functions ready.')

# %% [markdown] cell 12
# ## 5. Terrain, drainage, historical water, and built-up context

# %% cell 13
# Prefer the current Copernicus DEM release; fall back to the legacy collection if unavailable.
dem_collection_id = None
for candidate_id in ['COPERNICUS/DEM/GLO30_2024_1', 'COPERNICUS/DEM/GLO30']:
    try:
        candidate = ee.ImageCollection(candidate_id).filterBounds(study_aoi)
        if int(candidate.size().getInfo()) > 0:
            dem_collection_id = candidate_id
            break
    except Exception:
        continue
if dem_collection_id is None:
    raise RuntimeError('No Copernicus GLO-30 collection could be loaded.')

copdem_collection = ee.ImageCollection(dem_collection_id).filterBounds(study_aoi)
copdem_first = ee.Image(copdem_collection.first()).select('DEM')
dem = ee.Image(copdem_collection.mosaic()).select('DEM').rename('elevation_m').setDefaultProjection(copdem_first.projection())
terrain = ee.Terrain.products(dem)
slope = terrain.select('slope').rename('slope_deg')
aspect = terrain.select('aspect').rename('aspect_deg')
hillshade = terrain.select('hillshade').rename('hillshade')
roughness = dem.reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.circle(90, 'meters')).rename('roughness_90m')
tpi = dem.subtract(dem.focal_mean(150, 'circle', 'meters')).rename('tpi_150m')
terrain_stack = ee.Image.cat([dem, slope, aspect, roughness, tpi])

merit = ee.Image('MERIT/Hydro/v1_0_1')
upa = merit.select('upa').rename('upstream_area_km2')
hand = merit.select('hnd').rename('height_above_nearest_drainage_m')
main_drainage = upa.gte(MERIT_UPA_STREAM_KM2).selfMask().rename('main_drainage')
hydro_context = main_drainage.focal_max(750, 'circle', 'meters').unmask(0).toByte()
source_context = ee.Image.constant(1).clip(source_search_aoi).unmask(0).toByte().rename('source_context')

jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
water_occurrence = jrc.select('occurrence').unmask(0).rename('jrc_water_occurrence_pct')
permanent_water = water_occurrence.gte(50).rename('permanent_water')
seasonal_or_permanent_water = water_occurrence.gte(10).rename('historical_water')

worldcover = ee.Image(ee.ImageCollection('ESA/WorldCover/v200').first()).select('Map').rename('worldcover_2021')
built_context = worldcover.eq(50).rename('built_context')

print('DEM collection:', dem_collection_id)
print('Terrain and contextual layers ready.')

# %% cell 14
Map_terrain = geemap.Map(basemap='HYBRID')
Map_terrain.centerObject(study_aoi, 8)
Map_terrain.addLayer(hillshade, {'min': 0, 'max': 255}, 'Hillshade')
Map_terrain.addLayer(slope, {'min': 0, 'max': 70, 'palette': ['ffffff', 'fdae61', 'd73027']}, 'Slope (degrees)', False)
Map_terrain.addLayer(main_drainage, {'palette': ['00bfff']}, 'MERIT drainage, UPA >= threshold')
Map_terrain.addLayer(permanent_water.selfMask(), {'palette': ['0000ff']}, 'Historical persistent water', False)
Map_terrain.addLayer(built_context.selfMask(), {'palette': ['ff0000']}, 'WorldCover built-up context', False)
Map_terrain.addLayerControl()
Map_terrain

# %% [markdown] cell 15
# ## 6. Sentinel-2 inventory, cloud masking, and closest-valid-pixel composites
#
# Cloud probability is used when a matching `COPERNICUS/S2_CLOUD_PROBABILITY` image is available. The fallback masks only invalid, saturated, cloud-shadow, cloud, and cirrus SCL classes; it does not whitelist just four surface classes. The explicit `ee.Image(...)` cast after `copyProperties()` prevents the generic-`Element` error in the earlier notebook.
#
# **v1.1 schema correction:** every mapped Sentinel-2 image is explicitly cast to a fixed band order and PixelType. This prevents metadata-derived solar-angle constants from making the collection heterogeneous during `mosaic()`.

# %% cell 16
S2_COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED'
S2_CLOUD_COLLECTION_ID = 'COPERNICUS/S2_CLOUD_PROBABILITY'
S2_REFLECTANCE_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
S2_AUX_BANDS = ['SCL', 'solar_azimuth_deg', 'solar_zenith_deg']
S2_OUTPUT_BANDS = S2_REFLECTANCE_BANDS + S2_AUX_BANDS

# Earth Engine requires every image in an ImageCollection to have an identical
# band schema, including each band's numeric PixelType. Metadata-derived
# constant images otherwise inherit a scene-specific min/max range, e.g.
# Float<128.67, 128.67>, which makes mosaic() reject the collection.
S2_BAND_TYPES = {
    'B2': 'float',
    'B3': 'float',
    'B4': 'float',
    'B8': 'float',
    'B11': 'float',
    'B12': 'float',
    'SCL': 'uint8',
    'solar_azimuth_deg': 'float',
    'solar_zenith_deg': 'float',
}


def enforce_s2_schema(collection: ee.ImageCollection) -> ee.ImageCollection:
    """Return a homogeneous collection with fixed names, order, and PixelTypes."""
    collection = ee.ImageCollection(collection).select(S2_OUTPUT_BANDS)
    return ee.ImageCollection(collection.cast(S2_BAND_TYPES, S2_OUTPUT_BANDS))


def verified_s2_collection(ids: Sequence[str]) -> ee.ImageCollection:
    return ee.ImageCollection.fromImages([ee.Image(f'{S2_COLLECTION_ID}/{item}') for item in ids])


def load_raw_s2(ids: Sequence[str], phase: str) -> ee.ImageCollection:
    base = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(study_aoi)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', S2_SCENE_CLOUD_MAX))
    )
    if phase == 'pre':
        dynamic = (
            base.filterDate(S2_PRIMARY_PRE_START, S2_PRE_END)
            .merge(base.filterDate(S2_FALLBACK_PRE_START, S2_FALLBACK_PRE_END))
        )
    elif phase == 'post':
        dynamic = base.filterDate(S2_POST_START, S2_POST_END)
    else:
        raise ValueError("phase must be 'pre' or 'post'")

    if ADD_VERIFIED_S2_SCENES:
        try:
            dynamic = dynamic.merge(verified_s2_collection(ids))
        except Exception as exc:
            print(f'WARNING: verified {phase} scenes could not be merged: {exc}')
    return dynamic.distinct('system:index').sort('system:time_start')


def mask_scale_s2(image: ee.Image, use_cloud_probability: bool) -> ee.Image:
    """Mask and scale one Sentinel-2 image while enforcing a fixed band schema."""
    image = ee.Image(image)
    scl = image.select('SCL')
    bad_scl = (
        scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3))
        .Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    )
    edge_mask = image.select('B8A').mask().And(image.select('B9').mask())
    clear = bad_scl.Not().And(edge_mask)
    if use_cloud_probability:
        cloud_image = ee.Image(image.get('cloud_probability_image'))
        clear = clear.And(cloud_image.select('probability').lt(S2_CLOUD_PROBABILITY_MAX))

    # Fix each band's type explicitly. The .toFloat() calls on the solar-angle
    # constants are essential: without them their inferred min/max range differs
    # between scenes and the ImageCollection is not homogeneous.
    reflectance = image.select(S2_REFLECTANCE_BANDS).multiply(0.0001).toFloat()
    scl_band = scl.rename('SCL').toByte()

    reference_projection = image.select('B2').projection()
    solar_azimuth = (
        ee.Image.constant(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        .rename('solar_azimuth_deg')
        .toFloat()
        .setDefaultProjection(reference_projection)
    )
    solar_zenith = (
        ee.Image.constant(ee.Number(image.get('MEAN_SOLAR_ZENITH_ANGLE')))
        .rename('solar_zenith_deg')
        .toFloat()
        .setDefaultProjection(reference_projection)
    )

    output = ee.Image.cat([reflectance, scl_band, solar_azimuth, solar_zenith])
    output = output.select(S2_OUTPUT_BANDS).cast(S2_BAND_TYPES, S2_OUTPUT_BANDS)
    output = output.updateMask(clear)
    return ee.Image(output.copyProperties(image, image.propertyNames()))


def prepare_s2(raw: ee.ImageCollection, label: str) -> Tuple[ee.ImageCollection, str]:
    cloud = (
        ee.ImageCollection(S2_CLOUD_COLLECTION_ID)
        .filterBounds(study_aoi)
        .filterDate(S2_FALLBACK_PRE_START, S2_POST_END)
    )
    join = ee.Join.saveFirst('cloud_probability_image')
    joined = ee.ImageCollection(join.apply(
        primary=raw,
        secondary=cloud,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index'),
    ))
    matched = joined.filter(ee.Filter.notNull(['cloud_probability_image']))
    raw_count = int(raw.size().getInfo())
    matched_count = int(matched.size().getInfo())
    if raw_count == 0:
        raise RuntimeError(f'No Sentinel-2 {label} scenes intersect the study corridor.')

    if matched_count == raw_count:
        print(f'{label}: cloud probability matched all {raw_count} scenes.')
        prepared = ee.ImageCollection(matched.map(lambda img: mask_scale_s2(ee.Image(img), True)))
        prepared = enforce_s2_schema(prepared)
        return prepared.sort('system:time_start'), 'cloud_probability_plus_SCL'

    if matched_count > 0:
        unmatched = ee.ImageCollection(ee.Join.inverted().apply(
            primary=raw,
            secondary=cloud,
            condition=ee.Filter.equals(leftField='system:index', rightField='system:index'),
        ))
        prepared_matched = ee.ImageCollection(matched.map(lambda img: mask_scale_s2(ee.Image(img), True)))
        prepared_unmatched = ee.ImageCollection(unmatched.map(lambda img: mask_scale_s2(ee.Image(img), False)))
        prepared = enforce_s2_schema(prepared_matched.merge(prepared_unmatched))
        print(f'{label}: cloud probability matched {matched_count}/{raw_count}; SCL fallback retained the remaining scenes.')
        return prepared.sort('system:time_start'), 'hybrid_cloud_probability_and_SCL'

    print(f'{label}: no cloud-probability matches; using SCL-only fallback for {raw_count} scenes.')
    prepared = ee.ImageCollection(raw.map(lambda img: mask_scale_s2(ee.Image(img), False)))
    prepared = enforce_s2_schema(prepared)
    return prepared.sort('system:time_start'), 'SCL_fallback'


def s2_inventory(collection: ee.ImageCollection, region: ee.Geometry, label: str) -> pd.DataFrame:
    def to_feature(img):
        img = ee.Image(img)
        valid = valid_fraction_ee(img, region, 'B4', ANALYSIS_SCALE_M)
        return ee.Feature(None, {
            'phase': label,
            'date_utc': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss'),
            'system_index': img.get('system:index'),
            'product_id': img.get('PRODUCT_ID'),
            'mgrs_tile': img.get('MGRS_TILE'),
            'scene_cloud_pct': img.get('CLOUDY_PIXEL_PERCENTAGE'),
            'source_zone_valid_fraction': valid,
        })
    return geemap.ee_to_df(ee.FeatureCollection(collection.map(to_feature)))


s2_pre_raw = load_raw_s2(S2_PRE_IDS, 'pre')
s2_post_raw = load_raw_s2(S2_POST_IDS, 'post')
s2_pre_collection, s2_pre_mask_method = prepare_s2(s2_pre_raw, 'pre-event')
s2_post_collection, s2_post_mask_method = prepare_s2(s2_post_raw, 'post-event')

# Fail early here if Earth Engine ever changes a source schema. These calls also
# verify that the collection can be evaluated before expensive mosaics/tables.
s2_pre_schema = ee.Image(s2_pre_collection.first()).bandTypes().getInfo()
s2_post_schema = ee.Image(s2_post_collection.first()).bandTypes().getInfo()
print('Sentinel-2 fixed schema (pre):', s2_pre_schema)
print('Sentinel-2 fixed schema (post):', s2_post_schema)

s2_pre_inventory_df = s2_inventory(s2_pre_collection, source_search_aoi, 'pre')
s2_post_inventory_df = s2_inventory(s2_post_collection, source_search_aoi, 'post')
for df in (s2_pre_inventory_df, s2_post_inventory_df):
    if not df.empty:
        df['date_utc'] = pd.to_datetime(df['date_utc'], utc=True, errors='coerce')
        df.sort_values(['date_utc', 'mgrs_tile'], inplace=True)

print('Pre-event scenes:', len(s2_pre_inventory_df), '| mask:', s2_pre_mask_method)
display(s2_pre_inventory_df)
print('Post-event scenes:', len(s2_post_inventory_df), '| mask:', s2_post_mask_method)
display(s2_post_inventory_df)

# %% cell 17
s2_pre = closest_valid_pixel_mosaic(s2_pre_collection, EVENT_TIME_UTC, 'S2 pre closest-valid-pixel mosaic').clip(study_aoi)
s2_post = closest_valid_pixel_mosaic(s2_post_collection, EVENT_TIME_UTC, 'S2 post closest-valid-pixel mosaic').clip(study_aoi)

# Terrain illumination based on the source scene solar angles carried into each mosaic pixel.
def cosine_illumination(image: ee.Image) -> ee.Image:
    slope_rad = slope.multiply(math.pi / 180)
    aspect_rad = aspect.multiply(math.pi / 180)
    zenith_rad = image.select('solar_zenith_deg').multiply(math.pi / 180)
    azimuth_rad = image.select('solar_azimuth_deg').multiply(math.pi / 180)
    return (
        slope_rad.cos().multiply(zenith_rad.cos())
        .add(slope_rad.sin().multiply(zenith_rad.sin()).multiply(azimuth_rad.subtract(aspect_rad).cos()))
        .rename('cosine_illumination')
    )

illumination_pre = cosine_illumination(s2_pre)
illumination_post = cosine_illumination(s2_post)
optical_valid_pre = s2_pre.select('B4').mask().And(illumination_pre.gt(MIN_COSINE_ILLUMINATION)).rename('optical_valid_pre')
optical_valid_post = s2_post.select('B4').mask().And(illumination_post.gt(MIN_COSINE_ILLUMINATION)).rename('optical_valid_post')
optical_valid_both = optical_valid_pre.And(optical_valid_post).rename('optical_valid_both')

coverage_rows = []
for zone_name, geometry in [('source', source_search_aoi), ('full_corridor', study_aoi)]:
    coverage_rows.append({
        'zone': zone_name,
        'pre_valid_fraction': local_valid_fraction(s2_pre.updateMask(optical_valid_pre), geometry),
        'post_valid_fraction': local_valid_fraction(s2_post.updateMask(optical_valid_post), geometry),
        'paired_valid_fraction': local_valid_fraction(s2_post.updateMask(optical_valid_both), geometry),
    })
coverage_df = pd.DataFrame(coverage_rows)
display(coverage_df)

# %% cell 18
Map_s2 = geemap.Map(basemap='HYBRID')
Map_s2.centerObject(study_aoi, 8)
Map_s2.addLayer(s2_pre, {'bands': ['B4', 'B3', 'B2'], 'min': 0.02, 'max': 0.35, 'gamma': 1.1}, 'S2 pre RGB')
Map_s2.addLayer(s2_post, {'bands': ['B4', 'B3', 'B2'], 'min': 0.02, 'max': 0.35, 'gamma': 1.1}, 'S2 post RGB')
Map_s2.addLayer(optical_valid_both.selfMask(), {'palette': ['00ff00']}, 'Strict paired optical validity', False)
Map_s2.addLayer(optical_valid_both.unmask(0).Not().selfMask(), {'palette': ['808080']}, 'No strict paired optical observation', False)
Map_s2.addLayerControl()
Map_s2

# %% [markdown] cell 19
# ## 7. Optical indices and change metrics
#
# NDSI and MNDWI both use green and SWIR1 in their standard two-band form; they are stored with both names for transparent interpretation. Snow and water are separated using NIR reflectance, vegetation, slope, historical water, and SAR evidence rather than treating the two labels as independent spectral equations.

# %% cell 20
INDEX_BANDS = ['NDVI', 'NDSI', 'MNDWI', 'NDMI', 'NBR', 'BSI']
CHANGE_REFLECTANCE_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']


def add_optical_indices(image: ee.Image) -> ee.Image:
    ndvi = safe_normalized_difference(image, 'B8', 'B4', 'NDVI')
    green_swir = safe_normalized_difference(image, 'B3', 'B11', 'green_swir1_index')
    ndsi = green_swir.rename('NDSI')
    mndwi = green_swir.rename('MNDWI')
    ndmi = safe_normalized_difference(image, 'B8', 'B11', 'NDMI')
    nbr = safe_normalized_difference(image, 'B8', 'B12', 'NBR')
    bsi_num = image.select('B11').add(image.select('B4')).subtract(image.select('B8')).subtract(image.select('B2'))
    bsi_den = image.select('B11').add(image.select('B4')).add(image.select('B8')).add(image.select('B2'))
    bsi = bsi_num.divide(bsi_den.where(bsi_den.abs().lt(1e-6), 1e-6)).rename('BSI')
    return image.addBands([ndvi, ndsi, mndwi, ndmi, nbr, bsi])

s2_pre_idx = add_optical_indices(s2_pre)
s2_post_idx = add_optical_indices(s2_post)

index_delta = s2_post_idx.select(INDEX_BANDS).subtract(s2_pre_idx.select(INDEX_BANDS))
index_delta = index_delta.rename([f'd{name}' for name in INDEX_BANDS]).updateMask(optical_valid_both)

pre_spectrum = s2_pre_idx.select(CHANGE_REFLECTANCE_BANDS).max(0)
post_spectrum = s2_post_idx.select(CHANGE_REFLECTANCE_BANDS).max(0)
reflectance_delta = post_spectrum.subtract(pre_spectrum).rename([f'd{name}' for name in CHANGE_REFLECTANCE_BANDS])
spectral_rms = reflectance_delta.pow(2).reduce(ee.Reducer.mean()).sqrt().rename('spectral_rms')
dot_product = pre_spectrum.multiply(post_spectrum).reduce(ee.Reducer.sum())
pre_norm = pre_spectrum.pow(2).reduce(ee.Reducer.sum()).sqrt()
post_norm = post_spectrum.pow(2).reduce(ee.Reducer.sum()).sqrt()
spectral_angle = dot_product.divide(pre_norm.multiply(post_norm).max(1e-6)).clamp(-1, 1).acos().rename('spectral_angle_rad')

optical_change_stack = ee.Image.cat([reflectance_delta, index_delta, spectral_rms, spectral_angle]).updateMask(optical_valid_both)
print('Optical index and change stack ready.')

# %% [markdown] cell 21
# ## 8. Sentinel-1: matched pass and relative orbit
#
# Only a pass/orbit track with both pre- and post-event coverage is used. `COPERNICUS/S1_GRD` is preferred; the near-real-time linear-power `GRD_FLOAT` collection is converted to decibels only when needed. Same-track pairing reduces, but does not eliminate, terrain-related SAR artefacts.

# %% cell 22
def build_s1_base(collection_id: str, linear_power: bool) -> ee.ImageCollection:
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(study_aoi)
        .filterDate(S1_PRE_START, S1_POST_END)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    )
    def prepare(img):
        img = ee.Image(img)
        vv_vh = img.select(['VV', 'VH'])
        if linear_power:
            vv_vh = vv_vh.max(1e-8).log10().multiply(10)
        angle = img.select('angle')
        valid = vv_vh.select('VV').gt(-35).And(vv_vh.select('VH').gt(-42)).And(angle.gt(29)).And(angle.lt(46))
        output = vv_vh.updateMask(valid).addBands(angle.updateMask(valid))
        return ee.Image(output.copyProperties(img, img.propertyNames()))
    return ee.ImageCollection(collection.map(prepare))


def s1_inventory(collection: ee.ImageCollection) -> pd.DataFrame:
    def to_feature(img):
        img = ee.Image(img)
        return ee.Feature(None, {
            'date_utc': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss'),
            'system_index': img.get('system:index'),
            'orbit_pass': img.get('orbitProperties_pass'),
            'relative_orbit': img.get('relativeOrbitNumber_start'),
            'platform': img.get('platform_number'),
        })
    return geemap.ee_to_df(ee.FeatureCollection(collection.map(to_feature)))

S1_AVAILABLE = False
s1_collection_id = None
s1_linear_power = False
s1_base = None
s1_inventory_df = pd.DataFrame()
s1_track_candidates_df = pd.DataFrame()
selected_s1_pass = None
selected_s1_orbit = None
selected_s1_pre_time = None
selected_s1_post_time = None

for candidate_id, is_linear in [('COPERNICUS/S1_GRD', False), ('COPERNICUS/S1_GRD_FLOAT', True)]:
    try:
        candidate = build_s1_base(candidate_id, is_linear)
        inventory = s1_inventory(candidate)
        if inventory.empty:
            continue
        inventory['date_utc'] = pd.to_datetime(inventory['date_utc'], utc=True, errors='coerce')
        inventory = inventory.dropna(subset=['date_utc']).copy()
        event_ts = pd.Timestamp(EVENT_TIME_UTC)
        inventory['phase'] = np.where(inventory['date_utc'] < event_ts, 'pre', 'post')
        rows = []
        for (orbit_pass, relative_orbit), group in inventory.dropna(subset=['orbit_pass', 'relative_orbit']).groupby(['orbit_pass', 'relative_orbit']):
            pre = group[group['phase'] == 'pre']
            post = group[group['phase'] == 'post']
            if pre.empty or post.empty:
                continue
            closest_pre = pre['date_utc'].max()
            closest_post = post['date_utc'].min()
            rows.append({
                'orbit_pass': orbit_pass,
                'relative_orbit': int(relative_orbit),
                'pre_count': len(pre),
                'post_count': len(post),
                'closest_pre': closest_pre,
                'closest_post': closest_post,
                'gap_days': (closest_post - closest_pre).total_seconds() / 86400,
            })
        tracks = pd.DataFrame(rows)
        if tracks.empty:
            continue
        tracks['minimum_phase_count'] = tracks[['pre_count', 'post_count']].min(axis=1)
        tracks = tracks.sort_values(['gap_days', 'minimum_phase_count'], ascending=[True, False])
        choice = tracks.iloc[0]
        S1_AVAILABLE = True
        s1_collection_id = candidate_id
        s1_linear_power = is_linear
        s1_base = candidate
        s1_inventory_df = inventory.sort_values('date_utc')
        s1_track_candidates_df = tracks
        selected_s1_pass = str(choice['orbit_pass'])
        selected_s1_orbit = int(choice['relative_orbit'])
        selected_s1_pre_time = pd.Timestamp(choice['closest_pre'])
        selected_s1_post_time = pd.Timestamp(choice['closest_post'])
        break
    except Exception as exc:
        print(f'WARNING: {candidate_id} failed: {exc}')

empty_masked = ee.Image.constant(0).updateMask(ee.Image.constant(0)).clip(study_aoi)
if S1_AVAILABLE:
    selected_track = (
        s1_base.filter(ee.Filter.eq('orbitProperties_pass', selected_s1_pass))
        .filter(ee.Filter.eq('relativeOrbitNumber_start', selected_s1_orbit))
    )
    pre_window_start = (selected_s1_pre_time - pd.Timedelta(hours=12)).isoformat()
    pre_window_end = (selected_s1_pre_time + pd.Timedelta(hours=12)).isoformat()
    post_window_start = (selected_s1_post_time - pd.Timedelta(hours=12)).isoformat()
    post_window_end = (selected_s1_post_time + pd.Timedelta(hours=12)).isoformat()
    # Keep every granule/slice from the nearest matched overpass so a long, multi-tile corridor is not truncated by limit().
    s1_pre_collection = selected_track.filterDate(pre_window_start, pre_window_end)
    s1_post_collection = selected_track.filterDate(post_window_start, post_window_end)
    s1_pre = ee.Image(s1_pre_collection.select(['VV', 'VH']).median()).focal_median(30, 'circle', 'meters').rename(['VV_pre', 'VH_pre'])
    s1_post = ee.Image(s1_post_collection.select(['VV', 'VH']).median()).focal_median(30, 'circle', 'meters').rename(['VV_post', 'VH_post'])
    s1_valid = s1_pre.select('VV_pre').mask().And(s1_post.select('VV_post').mask()).rename('sar_valid')
    d_vv = s1_post.select('VV_post').subtract(s1_pre.select('VV_pre')).rename('dVV_db')
    d_vh = s1_post.select('VH_post').subtract(s1_pre.select('VH_pre')).rename('dVH_db')
    d_ratio = s1_post.select('VV_post').subtract(s1_post.select('VH_post')).subtract(
        s1_pre.select('VV_pre').subtract(s1_pre.select('VH_pre'))
    ).rename('dVV_minus_VH_db')
    sar_change_stack = ee.Image.cat([s1_pre, s1_post, d_vv, d_vh, d_ratio, s1_valid]).clip(study_aoi)
    sar_disturbance = d_vv.abs().gt(SAR_CHANGE_DB_THRESHOLD).Or(d_vh.abs().gt(SAR_CHANGE_DB_THRESHOLD)).And(s1_valid).rename('sar_disturbance')

    drainage_or_known_water = (
        hydro_context.unmask(0)
        .Or(seasonal_or_permanent_water.unmask(0).focal_max(300, 'circle', 'meters'))
        .Or(source_context.unmask(0))
    )
    low_backscatter_after = s1_post.select('VV_post').lt(SAR_WATER_VV_DB).And(s1_post.select('VH_post').lt(SAR_WATER_VH_DB))
    sar_new_water = (
        low_backscatter_after
        .And(d_vv.lt(SAR_WATER_DROP_DB))
        .And(slope.lt(MAX_WATER_SLOPE_DEG))
        .And(drainage_or_known_water)
        .And(permanent_water.unmask(0).Not())
        .And(s1_valid)
    )
    sar_new_water = connected_mask(sar_new_water, MIN_CONNECTED_PIXELS).rename('sar_new_water')
else:
    s1_pre = empty_masked.rename('VV_pre').addBands(empty_masked.rename('VH_pre'))
    s1_post = empty_masked.rename('VV_post').addBands(empty_masked.rename('VH_post'))
    d_vv = empty_masked.rename('dVV_db')
    d_vh = empty_masked.rename('dVH_db')
    d_ratio = empty_masked.rename('dVV_minus_VH_db')
    s1_valid = ee.Image.constant(0).clip(study_aoi).toByte().rename('sar_valid')
    sar_disturbance = ee.Image.constant(0).clip(study_aoi).toByte().rename('sar_disturbance')
    sar_new_water = ee.Image.constant(0).clip(study_aoi).toByte().rename('sar_new_water')
    sar_change_stack = ee.Image.cat([s1_pre, s1_post, d_vv, d_vh, d_ratio, s1_valid])

print('Sentinel-1 available:', S1_AVAILABLE)
if S1_AVAILABLE:
    print('Collection:', s1_collection_id, '| pass:', selected_s1_pass, '| relative orbit:', selected_s1_orbit)
    print('Nearest matched overpasses:', selected_s1_pre_time, '->', selected_s1_post_time)
    print('Granules retained:', int(s1_pre_collection.size().getInfo()), 'pre and', int(s1_post_collection.size().getInfo()), 'post')
    display(s1_track_candidates_df)
    display(s1_inventory_df)

# %% cell 23
Map_s1 = geemap.Map(basemap='HYBRID')
Map_s1.centerObject(study_aoi, 8)
if S1_AVAILABLE:
    Map_s1.addLayer(s1_pre.select('VV_pre'), {'min': -25, 'max': 0}, 'S1 pre VV')
    Map_s1.addLayer(s1_post.select('VV_post'), {'min': -25, 'max': 0}, 'S1 post VV')
    Map_s1.addLayer(d_vv, {'min': -5, 'max': 5, 'palette': ['0000ff', 'ffffff', 'ff0000']}, 'S1 dVV (dB)')
    Map_s1.addLayer(sar_new_water.selfMask(), {'palette': ['00ffff']}, 'SAR new-water candidate')
else:
    print('No matched Sentinel-1 pass/orbit was available in the frozen window; optical and Landsat outputs will still run.')
Map_s1.addLayerControl()
Map_s1

# %% [markdown] cell 24
# ## 9. Landsat 9 independent optical cross-check
#
# Landsat is not merged numerically with Sentinel-2 because its spectral response, acquisition time, and 30 m resolution differ. It is retained as an independent visual and directional check.

# %% cell 25
L9_COLLECTION_ID = 'LANDSAT/LC09/C02/T1_L2'
LANDSAT_AVAILABLE = False
landsat_inventory_df = pd.DataFrame()
l9_pre_idx = None
l9_post_idx = None
l9_change = None


def prepare_l9(image: ee.Image) -> ee.Image:
    image = ee.Image(image)
    qa = image.select('QA_PIXEL')
    clear = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    clear = clear.And(image.select('QA_RADSAT').eq(0))
    optical = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)
    optical = optical.rename(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).updateMask(clear)
    return ee.Image(optical.copyProperties(image, image.propertyNames()))

try:
    l9_raw = ee.ImageCollection(L9_COLLECTION_ID).filterBounds(study_aoi)
    l9_pre_candidates = closest_collection(l9_raw, LANDSAT_PRE_START, LANDSAT_PRE_END, EVENT_TIME_UTC, limit=1)
    l9_post_candidates = closest_collection(l9_raw, LANDSAT_POST_START, LANDSAT_POST_END, EVENT_TIME_UTC, limit=1)
    if int(l9_pre_candidates.size().getInfo()) > 0 and int(l9_post_candidates.size().getInfo()) > 0:
        l9_pre_anchor = ee.Image(l9_pre_candidates.first())
        l9_post_anchor = ee.Image(l9_post_candidates.first())
        pre_time = ee.Date(l9_pre_anchor.get('system:time_start'))
        post_time = ee.Date(l9_post_anchor.get('system:time_start'))
        l9_pre_swath = l9_raw.filterDate(pre_time.advance(-12, 'hour'), pre_time.advance(12, 'hour'))
        l9_post_swath = l9_raw.filterDate(post_time.advance(-12, 'hour'), post_time.advance(12, 'hour'))
        l9_pre_mosaic = ee.Image(ee.ImageCollection(l9_pre_swath.map(prepare_l9)).mosaic()).clip(study_aoi)
        l9_post_mosaic = ee.Image(ee.ImageCollection(l9_post_swath.map(prepare_l9)).mosaic()).clip(study_aoi)
        l9_pre_idx = add_optical_indices(l9_pre_mosaic)
        l9_post_idx = add_optical_indices(l9_post_mosaic)
        l9_valid_both = l9_pre_idx.select('B4').mask().And(l9_post_idx.select('B4').mask())
        l9_change = l9_post_idx.select(INDEX_BANDS).subtract(l9_pre_idx.select(INDEX_BANDS)).rename([f'L9_d{x}' for x in INDEX_BANDS]).updateMask(l9_valid_both)
        LANDSAT_AVAILABLE = True
        landsat_inventory_df = pd.DataFrame([
            {
                'phase': 'pre', 'date_utc': image_date_string(l9_pre_anchor),
                'granule_count': int(l9_pre_swath.size().getInfo()),
                'system_indices': safe_get_info(l9_pre_swath.aggregate_array('system:index'), []),
            },
            {
                'phase': 'post', 'date_utc': image_date_string(l9_post_anchor),
                'granule_count': int(l9_post_swath.size().getInfo()),
                'system_indices': safe_get_info(l9_post_swath.aggregate_array('system:index'), []),
            },
        ])
except Exception as exc:
    print('WARNING: Landsat cross-check unavailable:', exc)

print('Landsat 9 available:', LANDSAT_AVAILABLE)
display(landsat_inventory_df)

# %% [markdown] cell 26
# ## 10. Provisional physical masks and seven-class impact map
#
# The classification is deliberately transparent. It creates high-confidence candidate masks using paired optical change, SAR change, topography, drainage context, historical water, and connected components. It does not call cloud/shadow gaps “unchanged.” Infrastructure damage is not inferred from WorldCover alone; the output is an **exposure candidate** requiring high-resolution verification.

# %% cell 27
# Spectral state masks
snow_pre = (
    s2_pre_idx.select('NDSI').gt(NDSI_SNOW_THRESHOLD)
    .And(s2_pre_idx.select('B8').gt(SNOW_MIN_NIR))
    .And(s2_pre_idx.select('B3').gt(0.12))
    .And(optical_valid_pre)
).rename('snow_pre')
snow_post = (
    s2_post_idx.select('NDSI').gt(NDSI_SNOW_THRESHOLD)
    .And(s2_post_idx.select('B8').gt(SNOW_MIN_NIR))
    .And(s2_post_idx.select('B3').gt(0.12))
    .And(optical_valid_post)
).rename('snow_post')

optical_water_pre = (
    s2_pre_idx.select('MNDWI').gt(MNDWI_WATER_THRESHOLD)
    .And(s2_pre_idx.select('B8').lt(WATER_MAX_NIR))
    .And(s2_pre_idx.select('NDVI').lt(0.20))
    .And(slope.lt(MAX_WATER_SLOPE_DEG))
    .And(optical_valid_pre)
).rename('optical_water_pre')
optical_water_post = (
    s2_post_idx.select('MNDWI').gt(MNDWI_WATER_THRESHOLD)
    .And(s2_post_idx.select('B8').lt(WATER_MAX_NIR))
    .And(s2_post_idx.select('NDVI').lt(0.20))
    .And(slope.lt(MAX_WATER_SLOPE_DEG))
    .And(optical_valid_post)
).rename('optical_water_post')

vegetation_pre = s2_pre_idx.select('NDVI').gt(NDVI_VEGETATION_THRESHOLD).And(optical_valid_pre).rename('vegetation_pre')
vegetation_post = s2_post_idx.select('NDVI').gt(NDVI_VEGETATION_THRESHOLD).And(optical_valid_post).rename('vegetation_post')

bare_pre = (
    s2_pre_idx.select('NDVI').lt(NDVI_BARE_MAX)
    .And(s2_pre_idx.select('BSI').gt(BSI_BARE_MIN))
    .And(snow_pre.unmask(0).Not()).And(optical_water_pre.unmask(0).Not())
    .And(optical_valid_pre)
).rename('bare_pre')
bare_post = (
    s2_post_idx.select('NDVI').lt(NDVI_BARE_MAX)
    .And(s2_post_idx.select('BSI').gt(BSI_BARE_MIN))
    .And(snow_post.unmask(0).Not()).And(optical_water_post.unmask(0).Not())
    .And(optical_valid_post)
).rename('bare_post')

strong_optical_change = (
    spectral_rms.gt(SPECTRAL_RMS_THRESHOLD)
    .Or(spectral_angle.gt(SPECTRAL_ANGLE_THRESHOLD_RAD))
    .Or(index_delta.select('dNBR').abs().gt(DNBR_ABS_THRESHOLD))
    .Or(index_delta.select('dNDVI').lt(DNDVI_LOSS_THRESHOLD))
    .Or(index_delta.select('dNDSI').lt(DNDSI_LOSS_THRESHOLD))
).And(optical_valid_both).rename('strong_optical_change')

snow_loss = snow_pre.And(snow_post.unmask(0).Not()).And(optical_valid_both).rename('snow_loss')
snow_gain = snow_post.And(snow_pre.unmask(0).Not()).And(optical_valid_both).rename('snow_gain')
vegetation_loss = vegetation_pre.And(vegetation_post.unmask(0).Not()).And(optical_valid_both).rename('vegetation_loss')
vegetation_gain = vegetation_post.And(vegetation_pre.unmask(0).Not()).And(optical_valid_both).rename('vegetation_gain')

optical_water_gain = optical_water_post.And(optical_water_pre.unmask(0).Not()).And(optical_valid_both)
optical_water_loss = optical_water_pre.And(optical_water_post.unmask(0).Not()).And(optical_valid_both)
water_pre = permanent_water.unmask(0).Or(optical_water_pre.unmask(0)).rename('water_pre')
water_gain = connected_mask(optical_water_gain.unmask(0).Or(sar_new_water.unmask(0)), MIN_CONNECTED_PIXELS).rename('water_gain')
water_post = permanent_water.unmask(0).Or(optical_water_post.unmask(0)).Or(water_gain.unmask(0)).rename('water_post')
water_loss = connected_mask(optical_water_loss.unmask(0).And(permanent_water.unmask(0).Not()), MIN_CONNECTED_PIXELS).rename('water_loss')

# Change interpretation is limited to the source search zone or a drainage neighbourhood.
drainage_neighbourhood = hydro_context.focal_max(500, 'circle', 'meters').unmask(0)
disturbance_context = source_context.Or(drainage_neighbourhood).unmask(0)

debris_raw = (
    bare_post
    .And(strong_optical_change.unmask(0).Or(sar_disturbance.unmask(0)))
    .And(disturbance_context)
).Or(
    snow_loss.And(index_delta.select('dBSI').gt(0.04)).And(slope.gt(10)).And(source_context)
).And(water_post.unmask(0).Not()).And(snow_post.unmask(0).Not())
debris_candidate = connected_mask(debris_raw, MIN_CONNECTED_PIXELS).rename('debris_candidate')

disturbed_raw = (
    strong_optical_change.unmask(0).Or(sar_disturbance.unmask(0))
    .And(water_post.unmask(0).Not())
    .And(snow_post.unmask(0).Not())
    .And(vegetation_post.unmask(0).Not())
    .And(debris_candidate.unmask(0).Not())
    .And(disturbance_context)
)
disturbed_terrain = connected_mask(disturbed_raw, MIN_CONNECTED_PIXELS).rename('disturbed_terrain')

stable_exposed_rock = (
    bare_post.And(strong_optical_change.unmask(0).Not())
    .And(debris_candidate.unmask(0).Not()).And(disturbed_terrain.unmask(0).Not())
).rename('stable_exposed_rock')

built_exposure_candidate = (
    built_context.unmask(0)
    .And(water_gain.unmask(0).Or(debris_candidate.unmask(0)).Or(disturbed_terrain.unmask(0)).focal_max(30, 'circle', 'meters'))
).rename('built_exposure_candidate')

# Baseline class map: paired optical analysis uses this for transitions.
pre_class = ee.Image.constant(0).toByte()
pre_class = pre_class.where(bare_pre, 4).where(vegetation_pre, 3).where(built_context, 7).where(snow_pre, 1).where(water_pre, 2)
pre_class = pre_class.updateMask(optical_valid_pre.unmask(0).Or(permanent_water.unmask(0))).rename('pre_class')

# Integrated post map: SAR-only valid pixels may remain class 0, not falsely unchanged.
integrated_valid = optical_valid_post.unmask(0).Or(s1_valid.unmask(0)).Or(permanent_water.unmask(0)).rename('integrated_valid')
post_class = ee.Image.constant(0).toByte()
post_class = post_class.where(stable_exposed_rock, 4).where(vegetation_post, 3).where(built_context, 7).where(snow_post, 1).where(water_post, 2).where(debris_candidate, 5).where(disturbed_terrain, 6)
post_class = post_class.updateMask(integrated_valid).rename('post_class')

# Candidate source evidence score and polygons are calculated later.
source_evidence_score = ee.Image.cat([
    snow_loss.unmask(0).rename('e1'),
    index_delta.select('dNDSI').lt(DNDSI_LOSS_THRESHOLD).And(optical_valid_both).unmask(0).rename('e2'),
    index_delta.select('dBSI').gt(0.04).And(optical_valid_both).unmask(0).rename('e3'),
    strong_optical_change.unmask(0).rename('e4'),
    sar_disturbance.unmask(0).rename('e5'),
]).reduce(ee.Reducer.sum()).rename('source_evidence_score')
source_candidate_mask = connected_mask(
    source_evidence_score.gte(2).And(slope.gt(20)).And(source_context).And(water_post.unmask(0).Not()),
    MIN_CONNECTED_PIXELS,
).rename('source_candidate')

print('Provisional physical masks and class maps ready.')

# %% cell 28
Map_impact = geemap.Map(basemap='HYBRID')
Map_impact.centerObject(study_aoi, 8)
Map_impact.addLayer(s2_post, {'bands': ['B4', 'B3', 'B2'], 'min': 0.02, 'max': 0.35, 'gamma': 1.1}, 'Post-event RGB')
Map_impact.addLayer(post_class, {'min': 0, 'max': 7, 'palette': CLASS_PALETTE}, 'Provisional post-event classes')
Map_impact.addLayer(water_gain.selfMask(), {'palette': ['00ffff']}, 'New-water candidate')
Map_impact.addLayer(snow_loss.selfMask(), {'palette': ['7b3294']}, 'Snow/ice loss candidate', False)
Map_impact.addLayer(debris_candidate.selfMask(), {'palette': ['8c510a']}, 'Debris/mudflow candidate')
Map_impact.addLayer(disturbed_terrain.selfMask(), {'palette': ['ff7f00']}, 'Disturbed terrain candidate')
Map_impact.addLayer(built_exposure_candidate.selfMask(), {'palette': ['ff0000']}, 'Built-up exposure candidate')
Map_impact.addLayer(source_candidate_mask.selfMask(), {'palette': ['ff00ff']}, 'Source disturbance candidate')
Map_impact.addLayer(optical_valid_both.unmask(0).Not().selfMask(), {'palette': ['808080']}, 'No strict paired optical observation', False)
Map_impact.addLayer(reaches.style(color='ffffff', fillColor='00000000', width=1), {}, 'Analysis reaches', False)
Map_impact.addLayerControl()
Map_impact

# %% [markdown] cell 29
# ## 10A. Batch-materialise the result graph before statistics
#
# The post-event class image contains closest-valid Sentinel-2 mosaics, SAR change, terrain context and several connected-component operations. Interactive reductions recompute that complete graph and can time out even for one class in one reach. This cell exports a compact byte-valued result stack as six non-overlapping reach assets, polls the batch tasks, resumes them after a Colab reconnect, mosaics the completed assets, and replaces the dynamic result variables with asset-backed images. A configuration hash prevents stale assets from being reused after thresholds or geometry are changed.

# %% cell 30
# Batch-materialise the expensive post-event classification graph before any
# interactive area reductions. This cell is placed after the provisional
# masks/class maps and before Section 11.

RESULT_ASSET_FORCE_OVERWRITE = False
RESULT_ASSET_POLL_SECONDS = 20
RESULT_ASSET_VISIBILITY_POLL_SECONDS = 5
RESULT_ASSET_VISIBILITY_TIMEOUT_SECONDS = 300
RESULT_ASSET_SHARD_SIZE = 128
# None uses Earth Engine's default task scheduling and is required for
# non-commercial projects. Registered commercial projects may set 0..9999.
RESULT_ASSET_PRIORITY: Optional[int] = None
RESULT_CLASS_NODATA = 255

# Reuse the exact frozen scenarios later in the sensitivity section. Their
# binary outputs are persisted together with the central result masks so that
# Section 14 does not reconstruct the full optical/SAR graph interactively.
SENSITIVITY_SCENARIOS = {
    'conservative': {
        'rms': 0.10,
        'angle': 0.16,
        'dnbr': 0.18,
        'sar_db': 3.5,
        'score': 3,
    },
    'central': {
        'rms': SPECTRAL_RMS_THRESHOLD,
        'angle': SPECTRAL_ANGLE_THRESHOLD_RAD,
        'dnbr': DNBR_ABS_THRESHOLD,
        'sar_db': SAR_CHANGE_DB_THRESHOLD,
        'score': 2,
    },
    'liberal': {
        'rms': 0.06,
        'angle': 0.09,
        'dnbr': 0.10,
        'sar_db': 1.8,
        'score': 2,
    },
}


# The asset name includes a deterministic configuration signature. A change
# to the event windows, geometry, thresholds, scene audit IDs or mapping unit
# therefore creates a new asset instead of silently reusing stale pixels.
RESULT_CONFIG_PAYLOAD = {
    'event_time_utc': EVENT_TIME_UTC,
    's2_pre_ids': S2_PRE_IDS,
    's2_post_ids': S2_POST_IDS,
    'source_search_lon_lat': SOURCE_SEARCH_LON_LAT,
    'source_search_radius_m': SOURCE_SEARCH_RADIUS_M,
    'corridor_half_width_m': CORRIDOR_HALF_WIDTH_M,
    'analysis_scale_m': ANALYSIS_SCALE_M,
    'analysis_crs': OUTPUT_CRS,
    'thresholds': {
        'ndsi_snow': NDSI_SNOW_THRESHOLD,
        'snow_min_nir': SNOW_MIN_NIR,
        'mndwi_water': MNDWI_WATER_THRESHOLD,
        'water_max_nir': WATER_MAX_NIR,
        'ndvi_vegetation': NDVI_VEGETATION_THRESHOLD,
        'ndvi_bare_max': NDVI_BARE_MAX,
        'bsi_bare_min': BSI_BARE_MIN,
        'spectral_rms': SPECTRAL_RMS_THRESHOLD,
        'spectral_angle': SPECTRAL_ANGLE_THRESHOLD_RAD,
        'dnbr_abs': DNBR_ABS_THRESHOLD,
        'dndvi_loss': DNDVI_LOSS_THRESHOLD,
        'dndsi_loss': DNDSI_LOSS_THRESHOLD,
        'sar_change_db': SAR_CHANGE_DB_THRESHOLD,
        'minimum_connected_pixels': MIN_CONNECTED_PIXELS,
        'connected_max_size': MIN_CONNECTED_PIXELS,
    },
    'sensitivity': SENSITIVITY_SCENARIOS,
}
RESULT_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(RESULT_CONFIG_PAYLOAD, sort_keys=True, default=str).encode('utf-8')
).hexdigest()
RESULT_ASSET_VERSION = f'v1_4_2_{RESULT_CONFIG_SHA256[:12]}'
RESULT_ASSET_ROOT = f'projects/{EE_PROJECT}/assets'
RESULT_ASSET_PREFIX = (
    f'{RESULT_ASSET_ROOT}/'
    f'Lhende_2026_result_stack_{RESULT_ASSET_VERSION}'
)


def _storage_class_band(image: ee.Image, name: str) -> ee.Image:
    """Store valid class 0 separately from nodata by using byte value 255."""
    return (
        ee.Image(image)
        .unmask(RESULT_CLASS_NODATA)
        .toByte()
        .rename(name)
        .clip(study_aoi)
    )


def _storage_binary_band(image: ee.Image, name: str) -> ee.Image:
    """Store every binary result as an unmasked byte band containing 0 or 1."""
    return (
        ee.Image(image)
        .unmask(0)
        .neq(0)
        .toByte()
        .rename(name)
        .clip(study_aoi)
    )


# Build sensitivity masks once. connected_mask() uses a neighbourhood no
# larger than required by the minimum mapping unit.
sensitivity_dynamic_masks: Dict[str, ee.Image] = {}
for scenario_name, threshold in SENSITIVITY_SCENARIOS.items():
    scenario_changed = (
        spectral_rms.gt(threshold['rms'])
        .Or(spectral_angle.gt(threshold['angle']))
        .Or(index_delta.select('dNBR').abs().gt(threshold['dnbr']))
        .And(optical_valid_both)
    )

    if S1_AVAILABLE:
        scenario_sar_change = (
            d_vv.abs().gt(threshold['sar_db'])
            .Or(d_vh.abs().gt(threshold['sar_db']))
            .And(s1_valid)
        )
        scenario_changed = scenario_changed.unmask(0).Or(
            scenario_sar_change.unmask(0)
        )

    scenario_source = connected_mask(
        source_evidence_score.gte(threshold['score'])
        .And(scenario_changed.unmask(0))
        .And(slope.gt(20))
        .And(source_context),
        MIN_CONNECTED_PIXELS,
    )
    scenario_debris = connected_mask(
        bare_post
        .And(scenario_changed.unmask(0))
        .And(disturbance_context)
        .And(water_post.unmask(0).Not()),
        MIN_CONNECTED_PIXELS,
    )

    sensitivity_dynamic_masks[f'sens_{scenario_name}_source'] = scenario_source
    sensitivity_dynamic_masks[f'sens_{scenario_name}_debris'] = scenario_debris


RESULT_BINARY_IMAGES: Dict[str, ee.Image] = {
    'optical_valid_pre': optical_valid_pre,
    'optical_valid_post': optical_valid_post,
    'optical_valid_both': optical_valid_both,
    'sar_valid': s1_valid,
    'snow_pre': snow_pre,
    'snow_post': snow_post,
    'snow_loss': snow_loss,
    'snow_gain': snow_gain,
    'water_pre': water_pre,
    'water_post': water_post,
    'water_gain': water_gain,
    'water_loss': water_loss,
    'vegetation_pre': vegetation_pre,
    'vegetation_post': vegetation_post,
    'vegetation_loss': vegetation_loss,
    'debris_candidate': debris_candidate,
    'disturbed_terrain': disturbed_terrain,
    'built_exposure_candidate': built_exposure_candidate,
    'source_candidate': source_candidate_mask,
    **sensitivity_dynamic_masks,
}

RESULT_BAND_ORDER = [
    'pre_class',
    'post_class',
    'source_evidence_score',
    *RESULT_BINARY_IMAGES.keys(),
]

result_stack_dynamic = ee.Image.cat([
    _storage_class_band(pre_class, 'pre_class'),
    _storage_class_band(post_class, 'post_class'),
    (
        ee.Image(source_evidence_score)
        .unmask(0)
        .clamp(0, 255)
        .toByte()
        .rename('source_evidence_score')
        .clip(study_aoi)
    ),
    *[
        _storage_binary_band(image, name)
        for name, image in RESULT_BINARY_IMAGES.items()
    ],
]).select(RESULT_BAND_ORDER).toByte().set({
    'study_event': EVENT_LABEL,
    'result_asset_version': RESULT_ASSET_VERSION,
    'configuration_sha256': RESULT_CONFIG_SHA256,
    'analysis_scale_m': ANALYSIS_SCALE_M,
    'analysis_crs': OUTPUT_CRS,
    'class_nodata': RESULT_CLASS_NODATA,
    'minimum_connected_pixels': MIN_CONNECTED_PIXELS,
})


def _normalise_ee_error(exc: Exception) -> str:
    """Normalise typographic quotes and whitespace in an Earth Engine error."""
    return ' '.join(
        str(exc)
        .replace('’', "'")
        .replace('‘', "'")
        .casefold()
        .split()
    )


def _asset_name_from_record(record: Dict[str, Any]) -> str:
    """Read a canonical Cloud asset name from BASIC or FULL list output."""
    return str(record.get('name') or record.get('id') or '')


def _list_child_asset_ids(parent: str) -> set[str]:
    """
    List direct children of an Earth Engine folder/project root.

    This avoids probing a not-yet-created child with getAsset(), whose translated
    404 message is intentionally ambiguous with an access-denied response.
    """
    try:
        response = ee.data.listAssets({
            'parent': parent,
            'view': 'BASIC',
        })
    except Exception as exc:
        raise RuntimeError(
            f'Cannot list the Earth Engine asset destination {parent}. '
            f'Confirm that project {EE_PROJECT!r} is Earth Engine-enabled, '
            'that it has been added in the Code Editor Assets panel (or its '
            'asset root has been initialised), and that the authenticated '
            'Google account can access the project. '
            f'Server response: {exc}'
        ) from exc

    assets = response.get('assets', []) if isinstance(response, dict) else []
    return {
        name
        for name in (_asset_name_from_record(record) for record in assets)
        if name
    }


def _ee_asset_exists(asset_id: str) -> bool:
    """Return whether a direct child asset exists without issuing getAsset()."""
    parent = asset_id.rsplit('/', 1)[0]
    return asset_id in _list_child_asset_ids(parent)


def _find_active_export(description: str) -> Optional[ee.batch.Task]:
    """Return an already-running task after a Colab reconnect, when present."""
    for task in ee.batch.Task.list():
        status = task.status()
        if (
            status.get('description') == description
            and status.get('state') in {'READY', 'RUNNING'}
        ):
            return task
    return None


def _wait_for_export_tasks(tasks: Dict[str, ee.batch.Task]) -> None:
    """Poll batch tasks and surface the task's complete server-side error."""
    pending = dict(tasks)
    last_state: Dict[str, str] = {}

    while pending:
        for reach_id, task in list(pending.items()):
            status = task.status()
            state = str(status.get('state', 'UNKNOWN'))
            if last_state.get(reach_id) != state:
                print(f'  {reach_id}: {state}')
                last_state[reach_id] = state

            if state == 'COMPLETED':
                pending.pop(reach_id)
            elif state in {'FAILED', 'CANCELLED'}:
                error_message = (
                    status.get('error_message')
                    or status.get('error')
                    or status
                )
                raise RuntimeError(
                    f'Result-stack export failed for {reach_id}: '
                    f'{error_message}'
                )

        if pending:
            time.sleep(RESULT_ASSET_POLL_SECONDS)


def _wait_until_assets_visible(asset_ids: Dict[str, str]) -> None:
    """Confirm that completed result shards are listed before loading them."""
    pending = dict(asset_ids)
    deadline = time.monotonic() + RESULT_ASSET_VISIBILITY_TIMEOUT_SECONDS

    while pending:
        visible = _list_child_asset_ids(RESULT_ASSET_ROOT)
        for reach_id, asset_id in list(pending.items()):
            if asset_id in visible:
                print(f'  {reach_id}: asset readable')
                pending.pop(reach_id)

        if not pending:
            return

        if time.monotonic() >= deadline:
            raise RuntimeError(
                'The following completed result assets did not become readable '
                f'within {RESULT_ASSET_VISIBILITY_TIMEOUT_SECONDS} seconds: '
                + ', '.join(pending.values())
            )

        time.sleep(RESULT_ASSET_VISIBILITY_POLL_SECONDS)


# Export one compact byte stack per non-overlapping reach. This keeps each
# batch graph spatially bounded; R01 already contains the full source-search
# circle. Assets are resumed/skipped automatically after a Colab reconnect.
RESULT_REACH_IDS = [f'R{i + 1:02d}' for i in range(len(waypoints) - 1)]
RESULT_ASSET_IDS: Dict[str, str] = {
    reach_id: f'{RESULT_ASSET_PREFIX}_{reach_id}'
    for reach_id in RESULT_REACH_IDS
}

# One root listing both validates access and identifies completed shards.
visible_result_assets = _list_child_asset_ids(RESULT_ASSET_ROOT)
print(f'Earth Engine asset destination accessible: {RESULT_ASSET_ROOT}')
print('Export scheduling: Earth Engine default (no custom priority).')

pending_result_tasks: Dict[str, ee.batch.Task] = {}
for reach_id in RESULT_REACH_IDS:
    asset_id = RESULT_ASSET_IDS[reach_id]
    description = f'Lhende_result_stack_{RESULT_ASSET_VERSION}_{reach_id}'
    reach_geometry = ee.Feature(
        reaches.filter(ee.Filter.eq('zone_id', reach_id)).first()
    ).geometry()

    if asset_id in visible_result_assets and not RESULT_ASSET_FORCE_OVERWRITE:
        print(f'{reach_id}: using existing asset {asset_id}')
        continue

    active_task = _find_active_export(description)
    if active_task is not None:
        print(f'{reach_id}: resuming existing batch task.')
        pending_result_tasks[reach_id] = active_task
        continue

    export_kwargs: Dict[str, Any] = {
        'image': result_stack_dynamic.clip(reach_geometry),
        'description': description,
        'assetId': asset_id,
        'region': reach_geometry.bounds(100),
        'scale': ANALYSIS_SCALE_M,
        'crs': OUTPUT_CRS,
        'maxPixels': int(MAX_PIXELS),
        'shardSize': RESULT_ASSET_SHARD_SIZE,
        'pyramidingPolicy': {'.default': 'mode'},
        'overwrite': RESULT_ASSET_FORCE_OVERWRITE,
    }

    # IMPORTANT: Do not include the priority key for a non-commercial project.
    # Earth Engine then applies its normal default scheduling automatically.
    if RESULT_ASSET_PRIORITY is not None:
        priority_value = int(RESULT_ASSET_PRIORITY)
        if not 0 <= priority_value <= 9999:
            raise ValueError('RESULT_ASSET_PRIORITY must be None or an integer from 0 to 9999.')
        export_kwargs['priority'] = priority_value

    try:
        task = ee.batch.Export.image.toAsset(**export_kwargs)
        task.start()
    except Exception as exc:
        normalised_error = _normalise_ee_error(exc)

        # Defensive compatibility: if a custom priority was supplied for a
        # non-commercial project, rebuild and resubmit the task without that key.
        if (
            'priority' in export_kwargs
            and 'custom priority can only be set on tasks in registered commercial projects'
            in normalised_error
        ):
            print(
                f'{reach_id}: custom priority is unavailable for this project; '
                'retrying with Earth Engine default scheduling.'
            )
            export_kwargs.pop('priority', None)
            try:
                task = ee.batch.Export.image.toAsset(**export_kwargs)
                task.start()
            except Exception as retry_exc:
                exc = retry_exc
            else:
                pending_result_tasks[reach_id] = task
                print(f'{reach_id}: batch export submitted with default priority.')
                continue

        # Protect against a narrow race in which another completed task creates
        # the asset after the initial root listing but before this submission.
        refreshed_assets = _list_child_asset_ids(RESULT_ASSET_ROOT)
        if asset_id in refreshed_assets and not RESULT_ASSET_FORCE_OVERWRITE:
            print(f'{reach_id}: asset appeared during submission; reusing it.')
            visible_result_assets.add(asset_id)
            continue

        raise RuntimeError(
            f'Could not submit result-stack export for {reach_id}. '
            f'Destination: {asset_id}. Confirm that the authenticated account '
            f'has permission to create Earth Engine assets in project '
            f'{EE_PROJECT!r}. Server response: {exc}'
        ) from exc

    pending_result_tasks[reach_id] = task
    print(f'{reach_id}: batch export submitted.')

if pending_result_tasks:
    _wait_for_export_tasks(pending_result_tasks)

# Batch completion and asset-list visibility can be separated briefly, so
# confirm every shard through the parent folder before loading the mosaic.
_wait_until_assets_visible(RESULT_ASSET_IDS)

analysis_result_stack = (
    ee.ImageCollection.fromImages([
        ee.Image(RESULT_ASSET_IDS[reach_id])
        for reach_id in RESULT_REACH_IDS
    ])
    .mosaic()
    .select(RESULT_BAND_ORDER)
    .clip(study_aoi)
)

actual_result_bands = analysis_result_stack.bandNames().getInfo()
missing_bands = [
    name for name in RESULT_BAND_ORDER
    if name not in actual_result_bands
]
if missing_bands:
    raise RuntimeError(
        'Materialised result stack is missing band(s): '
        + ', '.join(missing_bands)
    )


def _load_stored_class(name: str) -> ee.Image:
    stored = analysis_result_stack.select(name).toByte()
    return stored.updateMask(stored.neq(RESULT_CLASS_NODATA)).rename(name)


def _load_stored_binary(name: str) -> ee.Image:
    return analysis_result_stack.select(name).eq(1).toByte().rename(name)


# Replace all expensive dynamic result images with asset-backed images. Every
# downstream area reduction, transition, vectorisation and result export now
# reads stored pixels rather than rebuilding the full multisensor graph.
pre_class = _load_stored_class('pre_class')
post_class = _load_stored_class('post_class')
source_evidence_score = analysis_result_stack.select(
    'source_evidence_score'
).toByte()

optical_valid_pre = _load_stored_binary('optical_valid_pre')
optical_valid_post = _load_stored_binary('optical_valid_post')
optical_valid_both = _load_stored_binary('optical_valid_both')
s1_valid = _load_stored_binary('sar_valid')
snow_pre = _load_stored_binary('snow_pre')
snow_post = _load_stored_binary('snow_post')
snow_loss = _load_stored_binary('snow_loss')
snow_gain = _load_stored_binary('snow_gain')
water_pre = _load_stored_binary('water_pre')
water_post = _load_stored_binary('water_post')
water_gain = _load_stored_binary('water_gain')
water_loss = _load_stored_binary('water_loss')
vegetation_pre = _load_stored_binary('vegetation_pre')
vegetation_post = _load_stored_binary('vegetation_post')
vegetation_loss = _load_stored_binary('vegetation_loss')
debris_candidate = _load_stored_binary('debris_candidate')
disturbed_terrain = _load_stored_binary('disturbed_terrain')
built_exposure_candidate = _load_stored_binary('built_exposure_candidate')
source_candidate_mask = _load_stored_binary('source_candidate')

sensitivity_asset_masks: Dict[str, ee.Image] = {
    f'{scenario_name}_source_km2': _load_stored_binary(
        f'sens_{scenario_name}_source'
    )
    for scenario_name in SENSITIVITY_SCENARIOS
}
sensitivity_asset_masks.update({
    f'{scenario_name}_debris_km2': _load_stored_binary(
        f'sens_{scenario_name}_debris'
    )
    for scenario_name in SENSITIVITY_SCENARIOS
})

print(
    'Materialised result stack ready:',
    len(RESULT_ASSET_IDS),
    'reach assets,',
    len(RESULT_BAND_ORDER),
    'byte bands.'
)

# %% [markdown] cell 31
# ## 11. Area statistics and pre/post transition matrix
# All class and binary masks entering this section are now read from the materialised Earth Engine assets rather than recomputed from the full multisensor graph.

# %% cell 32
# Asset-backed, timeout-safe and memory-safe area statistics.
# This block replaces the complete original Section 11 code cell.

AREA_SCALE_M = ANALYSIS_SCALE_M          # Preserve the planned 20 m analysis scale.
AREA_TILE_SCALE = 16                     # Maximum documented reduceRegion tileScale.
AREA_MAX_PIXELS = int(MAX_PIXELS)
AREA_BINARY_CHUNK_SIZE = 2               # Keep each fallback request small.
AREA_CRS = OUTPUT_CRS

# Asset-backed post-event and transition images are retained zone by zone for conservative execution.
AREA_TRY_COMBINED_PRE = True
AREA_TRY_COMBINED_POST = False
AREA_TRY_COMBINED_TRANSITION = False
AREA_TRY_COMBINED_BINARY = False

ZONE_COLUMNS = ['zone_id', 'zone_name', 'mid_km']
GROUPED_COLUMNS = ZONE_COLUMNS + ['group_id', 'area_km2']


def _is_ee_scaling_error(exc: Exception) -> bool:
    """
    Return True for recoverable Earth Engine interactive scaling failures.

    Earth Engine reports memory pressure and interactive request timeouts with
    different messages. Both require the same response here: avoid the combined
    request and evaluate smaller, sequential zone/band reductions.
    """
    text = str(exc).lower()
    markers = (
        # Memory / oversized aggregation
        'output of image computation is too large',
        'user memory limit exceeded',
        'memory limit',
        'try specifying a larger',
        'tilescale',
        '80.0 mib',
        # Interactive execution limit
        'computation timed out',
        'deadline exceeded',
        'request deadline',
        'timed out',
        # Related scaling/transient aggregation failures
        'too many concurrent aggregations',
        'service unavailable',
        'backend error',
        'http 503',
        'http 504',
    )
    return any(marker in text for marker in markers)


def _metadata_only_feature(feature: ee.Feature) -> ee.Feature:
    feature = ee.Feature(feature)
    return ee.Feature(None, {
        'zone_id': feature.get('zone_id'),
        'zone_name': feature.get('zone_name'),
        'mid_km': feature.get('mid_km'),
    })


def fetch_zone_metadata(zones: ee.FeatureCollection) -> pd.DataFrame:
    """Download only the small zone-property table; no raster is evaluated here."""
    metadata_fc = ee.FeatureCollection(zones.map(_metadata_only_feature)).sort('zone_id')
    info = metadata_fc.getInfo()
    rows = [feature.get('properties', {}) for feature in info.get('features', [])]
    frame = pd.DataFrame(rows, columns=ZONE_COLUMNS)
    if not frame.empty:
        frame['zone_id'] = frame['zone_id'].astype(str)
        frame['mid_km'] = pd.to_numeric(frame['mid_km'], errors='coerce')
    return frame


def dataframe_to_ee_fc(frame: pd.DataFrame) -> ee.FeatureCollection:
    """Create a small geometry-free FeatureCollection for later Drive-table exports."""
    if frame is None or frame.empty:
        return ee.FeatureCollection([])
    records = json.loads(frame.to_json(orient='records'))
    return ee.FeatureCollection([ee.Feature(None, record) for record in records])


def _normalise_grouped_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a stable schema for grouped class/transition area results."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=GROUPED_COLUMNS)
    frame = frame.copy()
    for column in GROUPED_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[GROUPED_COLUMNS]
    frame['zone_id'] = frame['zone_id'].astype(str)
    frame['group_id'] = pd.to_numeric(frame['group_id'], errors='coerce').astype('Int64')
    frame['area_km2'] = pd.to_numeric(frame['area_km2'], errors='coerce').fillna(0.0)
    frame['mid_km'] = pd.to_numeric(frame['mid_km'], errors='coerce')
    return frame.sort_values(['zone_id', 'group_id']).reset_index(drop=True)


def _append_full_zone_grouped(
    frame: pd.DataFrame,
    zone_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct Z_FULL by summing the non-overlapping Rxx reaches.

    This avoids evaluating the largest geometry again. The notebook's default
    reach construction partitions the full study corridor; Z_SOURCE is not
    included in this sum because it overlaps the first reach.
    """
    frame = _normalise_grouped_frame(frame)
    if frame.empty or (frame['zone_id'] == 'Z_FULL').any():
        return frame

    reach_rows = frame[frame['zone_id'].str.match(r'^R\d+$', na=False)].copy()
    if reach_rows.empty:
        return frame

    full_meta = zone_metadata[zone_metadata['zone_id'] == 'Z_FULL']
    full_zone_name = (
        str(full_meta.iloc[0]['zone_name'])
        if not full_meta.empty else 'Full study corridor'
    )
    full_mid_km = (
        float(full_meta.iloc[0]['mid_km'])
        if not full_meta.empty and pd.notna(full_meta.iloc[0]['mid_km'])
        else np.nan
    )

    full_rows = (
        reach_rows.groupby('group_id', dropna=False, as_index=False)['area_km2']
        .sum()
    )
    full_rows['zone_id'] = 'Z_FULL'
    full_rows['zone_name'] = full_zone_name
    full_rows['mid_km'] = full_mid_km
    full_rows = full_rows[GROUPED_COLUMNS]

    return _normalise_grouped_frame(pd.concat([frame, full_rows], ignore_index=True))


def flat_grouped_area(
    class_image: ee.Image,
    zones: ee.FeatureCollection,
    class_band: str,
) -> ee.FeatureCollection:
    """
    Efficient first attempt: grouped area for every zone with small aggregation tiles.
    """
    area_and_class = (
        ee.Image.pixelArea()
        .divide(1e6)
        .toDouble()
        .rename('area_km2')
        .addBands(ee.Image(class_image).rename(class_band).toInt16())
    )

    grouped = area_and_class.reduceRegions(
        collection=zones,
        reducer=ee.Reducer.sum().group(groupField=1, groupName='group_id'),
        scale=AREA_SCALE_M,
        crs=AREA_CRS,
        tileScale=AREA_TILE_SCALE,
        maxPixelsPerRegion=AREA_MAX_PIXELS,
    )

    features = grouped.toList(grouped.size())

    def expand(item):
        feature = ee.Feature(item)
        groups = ee.List(feature.toDictionary().get('groups', ee.List([])))

        def one(group):
            group = ee.Dictionary(group)
            return ee.Feature(None, {
                'zone_id': feature.get('zone_id'),
                'zone_name': feature.get('zone_name'),
                'mid_km': feature.get('mid_km'),
                'group_id': group.get('group_id'),
                'area_km2': group.get('sum'),
            })

        return ee.FeatureCollection(groups.map(one))

    return ee.FeatureCollection(features.map(expand)).flatten()


def _candidate_group_values(class_band: str) -> List[int]:
    """Known class domains used only by the low-memory fallback."""
    if class_band in {'pre_class', 'post_class'}:
        return list(range(8))
    if class_band == 'transition_code':
        return [pre_id * 10 + post_id for pre_id in range(8) for post_id in range(8)]
    raise ValueError(
        f'No safe fallback group domain is defined for {class_band!r}. '
        'Add its valid integer values to _candidate_group_values().'
    )


def _binary_area_image(items: Sequence[Tuple[str, ee.Image]]) -> ee.Image:
    """Build an unmasked area stack for a small set of binary masks."""
    pixel_area_km2 = ee.Image.pixelArea().divide(1e6).toDouble()
    bands = []
    for name, mask in items:
        binary = ee.Image(mask).unmask(0, False).neq(0).toDouble()
        bands.append(pixel_area_km2.multiply(binary).rename(name).toDouble())
    return ee.Image.cat(bands)


def _reduce_binary_items_one_zone(
    items: Sequence[Tuple[str, ee.Image]],
    zone_feature: ee.Feature,
    zone_id: str,
) -> Dict[str, float]:
    """
    Reduce one small mask stack over one zone. If needed, recursively split
    the stack until each request contains only one output band.
    """
    items = list(items)
    if not items:
        return {}

    stack = _binary_area_image(items)
    try:
        result = stack.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=zone_feature.geometry(),
            scale=AREA_SCALE_M,
            crs=AREA_CRS,
            bestEffort=False,
            maxPixels=AREA_MAX_PIXELS,
            tileScale=AREA_TILE_SCALE,
        ).getInfo()
        result = result or {}
        return {
            name: float(result.get(name) or 0.0)
            for name, _ in items
        }
    except Exception as exc:
        if not _is_ee_scaling_error(exc) or len(items) == 1:
            metric_names = ', '.join(name for name, _ in items)
            raise RuntimeError(
                f'Area reduction failed for zone {zone_id} and metric(s) '
                f'{metric_names}: {exc}'
            ) from exc

        midpoint = len(items) // 2
        left = _reduce_binary_items_one_zone(items[:midpoint], zone_feature, zone_id)
        right = _reduce_binary_items_one_zone(items[midpoint:], zone_feature, zone_id)
        return {**left, **right}


def _grouped_area_zone_by_zone(
    class_image: ee.Image,
    zones: ee.FeatureCollection,
    class_band: str,
    zone_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Low-memory fallback. It skips direct reduction of Z_FULL and reconstructs
    that row from the non-overlapping reaches.
    """
    area_and_class = (
        ee.Image.pixelArea()
        .divide(1e6)
        .toDouble()
        .rename('area_km2')
        .addBands(ee.Image(class_image).rename(class_band).toInt16())
    )

    rows: List[Dict[str, Any]] = []
    reduction_metadata = zone_metadata[zone_metadata['zone_id'] != 'Z_FULL'].copy()

    for index, meta in reduction_metadata.reset_index(drop=True).iterrows():
        zone_id = str(meta['zone_id'])
        print(
            f'  {class_band}: reducing {zone_id} '
            f'({index + 1}/{len(reduction_metadata)})'
        )
        zone_feature = ee.Feature(
            zones.filter(ee.Filter.eq('zone_id', zone_id)).first()
        )

        try:
            result = area_and_class.reduceRegion(
                reducer=ee.Reducer.sum().group(
                    groupField=1,
                    groupName='group_id',
                ),
                geometry=zone_feature.geometry(),
                scale=AREA_SCALE_M,
                crs=AREA_CRS,
                bestEffort=False,
                maxPixels=AREA_MAX_PIXELS,
                tileScale=AREA_TILE_SCALE,
            ).getInfo() or {}
            groups = result.get('groups') or []
        except Exception as exc:
            if not _is_ee_scaling_error(exc):
                raise

            # Last-resort exact fallback: calculate each valid code as a binary mask.
            print(f'    grouped reducer exceeded a memory/time limit for {zone_id}; using code-wise fallback.')
            group_values = _candidate_group_values(class_band)
            items = [
                (f'g_{value}', ee.Image(class_image).eq(value))
                for value in group_values
            ]
            binary_result = _reduce_binary_items_one_zone(items, zone_feature, zone_id)
            groups = [
                {'group_id': value, 'sum': binary_result.get(f'g_{value}', 0.0)}
                for value in group_values
                if binary_result.get(f'g_{value}', 0.0) > 0
            ]

        for group in groups:
            rows.append({
                'zone_id': zone_id,
                'zone_name': meta['zone_name'],
                'mid_km': meta['mid_km'],
                'group_id': group.get('group_id'),
                'area_km2': group.get('sum', 0.0),
            })

    frame = _normalise_grouped_frame(pd.DataFrame(rows, columns=GROUPED_COLUMNS))
    return _append_full_zone_grouped(frame, zone_metadata)


def evaluate_grouped_area(
    class_image: ee.Image,
    zones: ee.FeatureCollection,
    class_band: str,
    zone_metadata: pd.DataFrame,
    try_combined: bool = True,
) -> Tuple[pd.DataFrame, ee.FeatureCollection]:
    """
    Evaluate grouped areas with a timeout-aware fallback.

    For computationally expensive post-event and transition images, set
    try_combined=False so the code immediately uses smaller sequential zone
    reductions instead of first consuming an interactive request that is likely
    to time out.
    """
    if not try_combined:
        print(
            f'{class_band}: using zone-by-zone reduction from the outset '
            'to avoid an interactive timeout.'
        )
        frame = _grouped_area_zone_by_zone(
            class_image,
            zones,
            class_band,
            zone_metadata,
        )
        return frame, dataframe_to_ee_fc(frame)

    collection = flat_grouped_area(class_image, zones, class_band)
    try:
        frame = _normalise_grouped_frame(geemap.ee_to_df(collection))
        print(f'{class_band}: combined reduction completed with tileScale={AREA_TILE_SCALE}.')
        return frame, dataframe_to_ee_fc(frame)
    except Exception as exc:
        if not _is_ee_scaling_error(exc):
            raise
        print(
            f'{class_band}: combined request exceeded an interactive memory/time limit; '
            'switching to zone-by-zone reduction.'
        )
        frame = _grouped_area_zone_by_zone(
            class_image,
            zones,
            class_band,
            zone_metadata,
        )
        return frame, dataframe_to_ee_fc(frame)


def _binary_chunk_frame_zone_by_zone(
    items: Sequence[Tuple[str, ee.Image]],
    zones: ee.FeatureCollection,
    zone_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Low-memory fallback for one small chunk of binary masks."""
    names = [name for name, _ in items]
    rows: List[Dict[str, Any]] = []
    reduction_metadata = zone_metadata[zone_metadata['zone_id'] != 'Z_FULL'].copy()

    for _, meta in reduction_metadata.iterrows():
        zone_id = str(meta['zone_id'])
        zone_feature = ee.Feature(
            zones.filter(ee.Filter.eq('zone_id', zone_id)).first()
        )
        values = _reduce_binary_items_one_zone(items, zone_feature, zone_id)
        rows.append({
            'zone_id': zone_id,
            'zone_name': meta['zone_name'],
            'mid_km': meta['mid_km'],
            **values,
        })

    frame = pd.DataFrame(rows, columns=ZONE_COLUMNS + names)

    # Reconstruct the full corridor from the non-overlapping reach rows.
    reach_rows = frame[frame['zone_id'].astype(str).str.match(r'^R\d+$', na=False)]
    if not reach_rows.empty:
        full_meta = zone_metadata[zone_metadata['zone_id'] == 'Z_FULL']
        full_row: Dict[str, Any] = {
            'zone_id': 'Z_FULL',
            'zone_name': (
                str(full_meta.iloc[0]['zone_name'])
                if not full_meta.empty else 'Full study corridor'
            ),
            'mid_km': (
                float(full_meta.iloc[0]['mid_km'])
                if not full_meta.empty and pd.notna(full_meta.iloc[0]['mid_km'])
                else np.nan
            ),
        }
        for name in names:
            full_row[name] = pd.to_numeric(
                reach_rows[name], errors='coerce'
            ).fillna(0.0).sum()
        frame = pd.concat([frame, pd.DataFrame([full_row])], ignore_index=True)

    return frame


def evaluate_binary_area_stack(
    mask_dict: Dict[str, ee.Image],
    zones: ee.FeatureCollection,
    zone_metadata: pd.DataFrame,
    chunk_size: int = AREA_BINARY_CHUNK_SIZE,
    try_combined: bool = AREA_TRY_COMBINED_BINARY,
) -> Tuple[pd.DataFrame, ee.FeatureCollection]:
    """
    Evaluate binary areas in small band chunks and merge them by zone.

    When try_combined=False, every chunk is evaluated zone by zone. This is
    slower than one table request but avoids interactive computeFeatures
    timeouts for complex upstream masks.
    """
    items = list(mask_dict.items())
    result = zone_metadata.copy()

    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        names = [name for name, _ in chunk]
        print(
            f'Binary-area chunk {start // chunk_size + 1}/'
            f'{math.ceil(len(items) / chunk_size)}: {", ".join(names)}'
        )

        if not try_combined:
            print('  Using sequential zone reductions for this chunk.')
            chunk_frame = _binary_chunk_frame_zone_by_zone(
                chunk,
                zones,
                zone_metadata,
            )[['zone_id'] + names]
        else:
            stack = _binary_area_image(chunk)
            collection = stack.reduceRegions(
                collection=zones,
                reducer=ee.Reducer.sum(),
                scale=AREA_SCALE_M,
                crs=AREA_CRS,
                tileScale=AREA_TILE_SCALE,
                maxPixelsPerRegion=AREA_MAX_PIXELS,
            )

            try:
                chunk_frame = geemap.ee_to_df(collection)
                for column in names:
                    if column not in chunk_frame.columns:
                        chunk_frame[column] = 0.0
                chunk_frame = chunk_frame[['zone_id'] + names]
            except Exception as exc:
                if not _is_ee_scaling_error(exc):
                    raise
                print(
                    '  Combined chunk exceeded an interactive memory/time limit; '
                    'reducing this chunk zone by zone.'
                )
                chunk_frame = _binary_chunk_frame_zone_by_zone(
                    chunk,
                    zones,
                    zone_metadata,
                )[['zone_id'] + names]

        result = result.merge(chunk_frame, on='zone_id', how='left')

    metric_columns = list(mask_dict)
    for column in metric_columns:
        result[column] = pd.to_numeric(result[column], errors='coerce').fillna(0.0)

    result['mid_km'] = pd.to_numeric(result['mid_km'], errors='coerce')
    result = result.sort_values('zone_id').reset_index(drop=True)
    return result, dataframe_to_ee_fc(result)


# Retrieve only the zone attributes once.
zone_metadata_df = fetch_zone_metadata(all_zones)
if zone_metadata_df.empty:
    raise RuntimeError('No analysis zones were found.')
display(zone_metadata_df)

# Class areas.
pre_class_area_raw_df, pre_class_area_fc = evaluate_grouped_area(
    pre_class,
    all_zones,
    'pre_class',
    zone_metadata_df,
    try_combined=AREA_TRY_COMBINED_PRE,
)
post_class_area_raw_df, post_class_area_fc = evaluate_grouped_area(
    post_class,
    all_zones,
    'post_class',
    zone_metadata_df,
    try_combined=AREA_TRY_COMBINED_POST,
)


def prepare_class_area_df(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = _normalise_grouped_frame(frame)
    if frame.empty:
        return pd.DataFrame(columns=[
            'zone_id', 'zone_name', 'mid_km', 'area_km2',
            'period', 'class_id', 'class_name',
        ])
    frame = frame.copy()
    frame['period'] = period
    frame['class_id'] = frame['group_id'].astype('Int64')
    frame['class_name'] = frame['class_id'].map(CLASS_NAMES)
    return (
        frame.drop(columns=['group_id'])
        .sort_values(['zone_id', 'class_id'])
        .reset_index(drop=True)
    )


pre_class_area_df = prepare_class_area_df(pre_class_area_raw_df, 'pre')
post_class_area_df = prepare_class_area_df(post_class_area_raw_df, 'post')
class_area_long_df = pd.concat(
    [pre_class_area_df, post_class_area_df],
    ignore_index=True,
)
class_area_long_fc = dataframe_to_ee_fc(class_area_long_df)

if class_area_long_df.empty:
    class_area_comparison_df = pd.DataFrame()
else:
    identifiers = ['zone_id', 'zone_name', 'mid_km', 'class_id', 'class_name']
    class_area_comparison_df = (
        class_area_long_df
        .pivot_table(
            index=identifiers,
            columns='period',
            values='area_km2',
            aggfunc='sum',
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for period in ['pre', 'post']:
        if period not in class_area_comparison_df.columns:
            class_area_comparison_df[period] = 0.0
    class_area_comparison_df['absolute_change_km2'] = (
        class_area_comparison_df['post'] - class_area_comparison_df['pre']
    )
    class_area_comparison_df['percent_change_from_pre'] = np.where(
        class_area_comparison_df['pre'] > 0,
        100.0
        * class_area_comparison_df['absolute_change_km2']
        / class_area_comparison_df['pre'],
        np.nan,
    )
    class_area_comparison_df = class_area_comparison_df.sort_values(
        ['zone_id', 'class_id']
    ).reset_index(drop=True)

print('Pre-event class areas')
display(pre_class_area_df)
print('Post-event class areas')
display(post_class_area_df)
print('Pre-versus-post area comparison')
display(class_area_comparison_df)

# Binary change/validity areas. These are deliberately evaluated in small chunks.
change_masks = {
    'optical_valid_pre_km2': optical_valid_pre,
    'optical_valid_post_km2': optical_valid_post,
    'optical_valid_both_km2': optical_valid_both,
    'sar_valid_km2': s1_valid,
    'snow_pre_km2': snow_pre,
    'snow_post_km2': snow_post,
    'snow_loss_km2': snow_loss,
    'snow_gain_km2': snow_gain,
    'water_pre_km2': water_pre,
    'water_post_km2': water_post,
    'water_gain_km2': water_gain,
    'water_loss_km2': water_loss,
    'vegetation_pre_km2': vegetation_pre,
    'vegetation_post_km2': vegetation_post,
    'vegetation_loss_km2': vegetation_loss,
    'debris_candidate_km2': debris_candidate,
    'disturbed_terrain_km2': disturbed_terrain,
    'built_exposure_candidate_km2': built_exposure_candidate,
    'source_candidate_km2': source_candidate_mask,
}

change_area_df, change_area_fc = evaluate_binary_area_stack(
    change_masks,
    all_zones,
    zone_metadata_df,
    chunk_size=AREA_BINARY_CHUNK_SIZE,
    try_combined=AREA_TRY_COMBINED_BINARY,
)
metric_columns = list(change_masks)
display(change_area_df)

# Strict optical transition matrix; cloud/shadow gaps remain excluded.
transition = (
    pre_class.multiply(10)
    .add(post_class)
    .rename('transition_code')
    .updateMask(optical_valid_both)
    .toInt16()
)
transition_area_raw_df, transition_area_fc = evaluate_grouped_area(
    transition,
    all_zones,
    'transition_code',
    zone_metadata_df,
    try_combined=AREA_TRY_COMBINED_TRANSITION,
)

transition_area_df = _normalise_grouped_frame(transition_area_raw_df)
if not transition_area_df.empty:
    transition_area_df['transition_code'] = transition_area_df['group_id'].astype('Int64')
    transition_area_df['pre_class_id'] = (
        transition_area_df['transition_code'] // 10
    ).astype('Int64')
    transition_area_df['post_class_id'] = (
        transition_area_df['transition_code'] % 10
    ).astype('Int64')
    transition_area_df['pre_class_name'] = (
        transition_area_df['pre_class_id'].map(CLASS_NAMES)
    )
    transition_area_df['post_class_name'] = (
        transition_area_df['post_class_id'].map(CLASS_NAMES)
    )
    transition_area_df = (
        transition_area_df.drop(columns=['group_id'])
        .sort_values(['zone_id', 'pre_class_id', 'post_class_id'])
        .reset_index(drop=True)
    )

# Keep the export FeatureCollection aligned with the final parsed transition table.
transition_area_fc = dataframe_to_ee_fc(transition_area_df)
display(transition_area_df.head(100))

print(
    'Area statistics completed at '
    f'{AREA_SCALE_M} m in {AREA_CRS} with tileScale={AREA_TILE_SCALE}; '
    f'combined_post={AREA_TRY_COMBINED_POST}, combined_binary={AREA_TRY_COMBINED_BINARY}.'
)

# %% [markdown] cell 33
# ## 12. Downstream propagation profile
#
# Reaches are ordered from the source to approximately 100 km downstream. The plotted values are candidate affected area inside each non-overlapping buffered reach, not discharge or travel-time estimates.

# %% cell 34
reach_change_df = change_area_df[change_area_df['zone_id'].astype(str).str.startswith('R')].copy()
if not reach_change_df.empty:
    reach_change_df['mid_km'] = pd.to_numeric(reach_change_df['mid_km'], errors='coerce')
    reach_change_df = reach_change_df.sort_values('mid_km')
    display(reach_change_df[['zone_id', 'zone_name', 'mid_km', 'water_gain_km2', 'debris_candidate_km2', 'disturbed_terrain_km2']])

    plt.figure(figsize=(10, 5))
    for column in ['water_gain_km2', 'debris_candidate_km2', 'disturbed_terrain_km2']:
        plt.plot(reach_change_df['mid_km'], reach_change_df[column], marker='o', label=column.replace('_km2', '').replace('_', ' '))
    plt.xlabel('Approximate distance from source (km)')
    plt.ylabel('Candidate affected area within reach (km²)')
    plt.title('Longitudinal propagation of mapped disturbance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print('No reach summary rows were returned.')

# %% [markdown] cell 35
# ## 13. Candidate source, temporary-water, debris, and disturbed-surface polygons

# %% cell 36
def vectorise_mask(mask: ee.Image, region: ee.Geometry, label: str, minimum_area_m2: float) -> ee.FeatureCollection:
    vectors = mask.unmask(0).toByte().selfMask().reduceToVectors(
        geometry=region,
        scale=ANALYSIS_SCALE_M,
        geometryType='polygon',
        eightConnected=True,
        labelProperty='mask_value',
        reducer=ee.Reducer.countEvery(),
        maxPixels=MAX_PIXELS,
        tileScale=TILE_SCALE,
    )
    def add_attributes(feature):
        feature = ee.Feature(feature)
        geom = feature.geometry()
        return feature.set({
            'candidate_type': label,
            'area_m2': geom.area(1),
            'perimeter_m': geom.perimeter(1),
        })
    return ee.FeatureCollection(vectors.map(add_attributes)).filter(ee.Filter.gte('area_m2', minimum_area_m2))

source_candidate_vectors = vectorise_mask(source_candidate_mask, source_search_aoi, 'source_disturbance', MIN_SOURCE_VECTOR_AREA_M2)
temporary_water_vectors = vectorise_mask(water_gain.And(source_context), source_search_aoi, 'temporary_water', MIN_VECTOR_AREA_M2)
debris_vectors = vectorise_mask(debris_candidate, study_aoi, 'debris_mudflow', MIN_VECTOR_AREA_M2)
disturbance_vectors = vectorise_mask(disturbed_terrain, study_aoi, 'disturbed_terrain', MIN_VECTOR_AREA_M2)

source_vector_count = int(safe_get_info(source_candidate_vectors.size(), 0, 'source vectors'))
temporary_water_count = int(safe_get_info(temporary_water_vectors.size(), 0, 'temporary-water vectors'))
debris_vector_count = int(safe_get_info(debris_vectors.size(), 0, 'debris vectors'))
disturbance_vector_count = int(safe_get_info(disturbance_vectors.size(), 0, 'disturbed-terrain vectors'))

source_candidate_df = geemap.ee_to_df(source_candidate_vectors) if source_vector_count else pd.DataFrame()
temporary_water_df = geemap.ee_to_df(temporary_water_vectors) if temporary_water_count else pd.DataFrame()

print('Source candidate polygons:', source_vector_count)
print('Temporary-water candidate polygons:', temporary_water_count)
print('Debris/mudflow candidate polygons:', debris_vector_count)
print('Disturbed-terrain candidate polygons:', disturbance_vector_count)
display(source_candidate_df)
display(temporary_water_df)

largest_source_candidate = None
largest_source_centroid = None
if source_vector_count:
    largest_source_candidate = ee.Feature(source_candidate_vectors.sort('area_m2', False).first())
    largest_source_centroid = largest_source_candidate.geometry().centroid(1)
    centroid_coordinates = safe_get_info(largest_source_centroid.coordinates(), [])
    print('Largest candidate centroid [longitude, latitude]:', centroid_coordinates)

Map_vectors = geemap.Map(basemap='HYBRID')
Map_vectors.centerObject(source_search_aoi, 11)
Map_vectors.addLayer(s2_post, {'bands': ['B4', 'B3', 'B2'], 'min': 0.02, 'max': 0.35}, 'Post-event RGB')
Map_vectors.addLayer(source_candidate_vectors.style(color='ff00ff', fillColor='ff00ff55'), {}, 'Source disturbance polygons')
Map_vectors.addLayer(temporary_water_vectors.style(color='00ffff', fillColor='00ffff55'), {}, 'Temporary-water polygons')
if largest_source_centroid is not None:
    Map_vectors.addLayer(ee.FeatureCollection([ee.Feature(largest_source_centroid)]).style(color='ffffff', pointSize=8), {}, 'Largest source candidate centroid')
Map_vectors.addLayerControl()
Map_vectors

# %% [markdown] cell 37
# ## 14. Threshold sensitivity analysis
#
# The primary quantities are recalculated under conservative, central, and liberal threshold sets. A result that appears only under the liberal thresholds should not be presented as robust.
# The six scenario masks were computed during the batch materialisation stage; this section performs only asset-backed area sums.

# %% cell 38
# Sensitivity masks were computed once in the batch result-stack exports and
# are now asset-backed.  Only lightweight pixel-area sums are evaluated here.
sensitivity_zone_metadata_df = fetch_zone_metadata(summary_zones)
if sensitivity_zone_metadata_df.empty:
    raise RuntimeError('No sensitivity-summary zones were found.')

sensitivity_rows: List[Dict[str, Any]] = []
sensitivity_items = list(sensitivity_asset_masks.items())
for _, meta in sensitivity_zone_metadata_df.sort_values('zone_id').iterrows():
    zone_id = str(meta['zone_id'])
    zone_feature = ee.Feature(
        summary_zones.filter(ee.Filter.eq('zone_id', zone_id)).first()
    )
    print(f'Sensitivity areas: reducing {zone_id}.')
    values = _reduce_binary_items_one_zone(
        sensitivity_items,
        zone_feature,
        zone_id,
    )
    sensitivity_rows.append({
        'zone_id': zone_id,
        'zone_name': meta['zone_name'],
        'mid_km': meta['mid_km'],
        **values,
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)
for column in sensitivity_asset_masks:
    sensitivity_df[column] = pd.to_numeric(
        sensitivity_df.get(column),
        errors='coerce',
    ).fillna(0.0)
sensitivity_df['mid_km'] = pd.to_numeric(
    sensitivity_df['mid_km'],
    errors='coerce',
)
sensitivity_df = sensitivity_df.sort_values('zone_id').reset_index(drop=True)
sensitivity_fc = dataframe_to_ee_fc(sensitivity_df)
display(sensitivity_df)

# %% [markdown] cell 39
# ## 15. Optional supervised Random Forest classification
#
# Create independent polygons with integer `class_id` values 1–7. Prefer a text field `split` with values `train` and `validation`. Without it, the notebook performs a seeded polygon-level 70/30 split and stops if any class is absent from either partition. Never create validation polygons from the provisional rule map.

# %% cell 40
predictor_parts = [
    prefix_bands(s2_pre_idx.select(S2_REFLECTANCE_BANDS + INDEX_BANDS), 'pre_'),
    prefix_bands(s2_post_idx.select(S2_REFLECTANCE_BANDS + INDEX_BANDS), 'post_'),
    optical_change_stack,
    terrain_stack,
    worldcover.toFloat(),
    water_occurrence.toFloat(),
]
if S1_AVAILABLE:
    predictor_parts.append(sar_change_stack.select(['VV_pre', 'VH_pre', 'VV_post', 'VH_post', 'dVV_db', 'dVH_db', 'dVV_minus_VH_db']))
predictor_stack = ee.Image.cat(predictor_parts).float().clip(study_aoi)

rf_classified = None
rf_validation_df = pd.DataFrame()
rf_feature_importance_df = pd.DataFrame()

if RUN_SUPERVISED_RF:
    labelled = ee.FeatureCollection(CUSTOM_TRAINING_ASSET).filterBounds(study_aoi)
    polygon_count = int(safe_get_info(labelled.size(), 0, 'labelled polygon count'))
    if polygon_count < 14:
        raise ValueError('Provide at least 14 labelled polygons and at least two spatially separated polygons per class.')
    class_hist = safe_get_info(labelled.aggregate_histogram('class_id'), {}, 'class histogram') or {}
    if any(int(v) < 2 for v in class_hist.values()):
        raise ValueError(f'Each class needs at least two polygons: {class_hist}')

    property_names = safe_get_info(ee.Feature(labelled.first()).propertyNames(), [], 'training properties') or []
    if RF_SPLIT_PROPERTY in property_names:
        train_polygons = labelled.filter(ee.Filter.eq(RF_SPLIT_PROPERTY, RF_TRAIN_VALUE))
        validation_polygons = labelled.filter(ee.Filter.eq(RF_SPLIT_PROPERTY, RF_VALIDATION_VALUE))
        split_method = f'explicit polygon split using {RF_SPLIT_PROPERTY}'
    else:
        labelled = labelled.randomColumn('polygon_random', RANDOM_SEED)
        train_polygons = labelled.filter(ee.Filter.lt('polygon_random', 0.70))
        validation_polygons = labelled.filter(ee.Filter.gte('polygon_random', 0.70))
        split_method = 'seeded 70/30 polygon-level split'

    train_hist = safe_get_info(train_polygons.aggregate_histogram('class_id'), {}, 'train class histogram') or {}
    valid_hist = safe_get_info(validation_polygons.aggregate_histogram('class_id'), {}, 'validation class histogram') or {}
    all_classes = {str(k) for k in class_hist}
    missing_train = sorted(all_classes - {str(k) for k in train_hist})
    missing_valid = sorted(all_classes - {str(k) for k in valid_hist})
    if missing_train or missing_valid:
        raise ValueError(f'Every class must occur in both partitions. Missing train={missing_train}; validation={missing_valid}. Add an explicit split field.')

    train_samples = predictor_stack.sampleRegions(train_polygons, ['class_id'], ANALYSIS_SCALE_M, geometries=False, tileScale=TILE_SCALE)
    validation_samples = predictor_stack.sampleRegions(validation_polygons, ['class_id'], ANALYSIS_SCALE_M, geometries=False, tileScale=TILE_SCALE)
    if int(train_samples.size().getInfo()) == 0 or int(validation_samples.size().getInfo()) == 0:
        raise ValueError('No valid predictor pixels were sampled; move polygons away from cloud/shadow gaps.')

    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=400, minLeafPopulation=2, bagFraction=0.7, seed=RANDOM_SEED
    ).train(train_samples, 'class_id', predictor_stack.bandNames())
    rf_classified = predictor_stack.classify(classifier).rename('rf_class').toByte()
    validated = validation_samples.classify(classifier)
    class_order = ee.List(labelled.aggregate_array('class_id')).distinct().sort()
    error_matrix = validated.errorMatrix('class_id', 'classification', class_order)
    rf_validation_df = pd.DataFrame([{
        'split_method': split_method,
        'labelled_polygons': polygon_count,
        'train_polygon_histogram': train_hist,
        'validation_polygon_histogram': valid_hist,
        'overall_accuracy': safe_get_info(error_matrix.accuracy()),
        'kappa': safe_get_info(error_matrix.kappa()),
        'class_order': safe_get_info(class_order, []),
        'confusion_matrix': safe_get_info(error_matrix.array(), []),
        'producers_accuracy': safe_get_info(error_matrix.producersAccuracy(), []),
        'consumers_accuracy': safe_get_info(error_matrix.consumersAccuracy(), []),
    }])
    importance = safe_get_info(ee.Dictionary(classifier.explain()).get('importance'), {}, 'RF importance') or {}
    rf_feature_importance_df = pd.DataFrame([{'predictor': k, 'importance': v} for k, v in importance.items()])
    if not rf_feature_importance_df.empty:
        rf_feature_importance_df = rf_feature_importance_df.sort_values('importance', ascending=False)
    display(rf_validation_df)
    display(rf_feature_importance_df.head(30))
else:
    print('Random Forest skipped. Set CUSTOM_TRAINING_ASSET to enable it.')

# %% [markdown] cell 41
# ## 16. Optional event-specific high-resolution open-data inventory
#
# These community-catalogued Vantor and Planet collections are useful for visual validation, but their licensing, cloud masks, radiometric products, footprints, and resolutions differ. The core analysis does not depend on them.

# %% cell 42
OPTIONAL_EVENT_COLLECTIONS = {
    'Vantor': 'projects/sat-io/open-datasets/VANTOR-DISASTER-DATA/NEPAL_FLOOD_2026',
    'Planet': 'projects/sat-io/open-datasets/PLANET-DISASTER-DATA/NEPAL_FLOOD_2026',
}
high_res_inventory_rows = []
if RUN_OPTIONAL_HIGH_RES_INVENTORY:
    for provider, asset_id in OPTIONAL_EVENT_COLLECTIONS.items():
        try:
            collection = ee.ImageCollection(asset_id).filterBounds(study_aoi)
            count = int(collection.size().getInfo())
            first = ee.Image(collection.first()) if count else None
            high_res_inventory_rows.append({
                'provider': provider,
                'asset_id': asset_id,
                'scene_count': count,
                'first_scene_bands': safe_get_info(first.bandNames(), []) if first else [],
                'first_scene_properties': safe_get_info(first.propertyNames(), []) if first else [],
            })
        except Exception as exc:
            high_res_inventory_rows.append({'provider': provider, 'asset_id': asset_id, 'scene_count': 0, 'error': str(exc)})
high_res_inventory_df = pd.DataFrame(high_res_inventory_rows)
display(high_res_inventory_df)

# %% [markdown] cell 43
# ## 17. Optional comparison with a CEMS EMSR927 reference asset
#
# Download the official EMSR927 vector products, inspect their licence/metadata, and upload the relevant observed-event layer as an Earth Engine FeatureCollection. The code reports overlap only; it does not treat emergency mapping as field truth.

# %% cell 44
cems_comparison_df = pd.DataFrame()
if CEMS_REFERENCE_ASSET:
    cems_reference_fc = ee.FeatureCollection(CEMS_REFERENCE_ASSET).filterBounds(study_aoi)
    reference_mask = ee.Image.constant(0).toByte().paint(cems_reference_fc, 1).rename('reference')
    detected_mask = water_gain.unmask(0).Or(debris_candidate.unmask(0)).Or(disturbed_terrain.unmask(0)).toByte().rename('detected')
    comparison = ee.Image.cat([
        reference_mask,
        detected_mask,
        reference_mask.And(detected_mask).rename('intersection'),
        reference_mask.And(detected_mask.Not()).rename('reference_only'),
        detected_mask.And(reference_mask.Not()).rename('detected_only'),
    ])
    cems_masks = {
        'reference_km2': comparison.select('reference'),
        'detected_km2': comparison.select('detected'),
        'intersection_km2': comparison.select('intersection'),
        'reference_only_km2': comparison.select('reference_only'),
        'detected_only_km2': comparison.select('detected_only'),
    }

    # Only two summary zones are involved, so reduce each one sequentially.
    cems_zone_metadata = fetch_zone_metadata(summary_zones)
    cems_items = list(cems_masks.items())
    cems_rows = []
    for _, meta in cems_zone_metadata.iterrows():
        zone_id = str(meta['zone_id'])
        zone_feature = ee.Feature(
            summary_zones.filter(ee.Filter.eq('zone_id', zone_id)).first()
        )
        values = _reduce_binary_items_one_zone(
            cems_items,
            zone_feature,
            zone_id,
        )
        cems_rows.append({
            'zone_id': zone_id,
            'zone_name': meta['zone_name'],
            'mid_km': meta['mid_km'],
            **values,
        })

    cems_comparison_df = pd.DataFrame(cems_rows)
    cems_area_fc = dataframe_to_ee_fc(cems_comparison_df)
    display(cems_comparison_df)
else:
    print('CEMS comparison skipped. Set CEMS_REFERENCE_ASSET after importing the chosen EMSR927 layer.')

# %% [markdown] cell 45
# ## 18. Export rasters and vectors to Google Drive
#
# Set `START_DRIVE_EXPORTS = True`, rerun this cell, then monitor the printed task IDs in the Earth Engine Tasks panel. Raster outputs use Cloud-Optimized GeoTIFF. Vector outputs use GeoJSON; statistics use CSV.

# %% cell 46
def start_image_export(image: ee.Image, description: str, scale: int = ANALYSIS_SCALE_M, nodata: float = -9999):
    export_image = image.unmask(nodata)
    task = ee.batch.Export.image.toDrive(
        image=export_image,
        description=description,
        folder=DRIVE_EXPORT_FOLDER,
        fileNamePrefix=description,
        region=study_aoi,
        scale=scale,
        crs=OUTPUT_CRS,
        maxPixels=MAX_PIXELS,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True, 'noData': nodata},
    )
    task.start()
    return task


def start_table_export(collection: ee.FeatureCollection, description: str, file_format: str = 'CSV'):
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=DRIVE_EXPORT_FOLDER,
        fileNamePrefix=description,
        fileFormat=file_format,
    )
    task.start()
    return task

export_tasks = []
if START_DRIVE_EXPORTS:
    raster_exports = {
        'R01_S2_pre_reflectance_indices': s2_pre_idx.select(S2_REFLECTANCE_BANDS + INDEX_BANDS),
        'R02_S2_post_reflectance_indices': s2_post_idx.select(S2_REFLECTANCE_BANDS + INDEX_BANDS),
        'R03_optical_change_metrics': optical_change_stack,
        'R04_terrain_hydro_context': ee.Image.cat([terrain_stack, upa, hand, water_occurrence, worldcover]),
        'R05_pre_class': pre_class,
        'R06_post_class_provisional': post_class,
        'R07_change_masks': ee.Image.cat(list(change_masks.values())).toByte(),
        'R08_source_evidence_score': source_evidence_score.toByte(),
        'R09_optical_validity': ee.Image.cat([optical_valid_pre, optical_valid_post, optical_valid_both]).toByte(),
    }
    if S1_AVAILABLE:
        raster_exports['R10_Sentinel1_change'] = sar_change_stack
    if LANDSAT_AVAILABLE:
        raster_exports['R11_Landsat9_index_change'] = l9_change
    if rf_classified is not None:
        raster_exports['R12_RF_post_class'] = rf_classified

    categorical = {'R05_pre_class', 'R06_post_class_provisional', 'R07_change_masks', 'R08_source_evidence_score', 'R09_optical_validity', 'R12_RF_post_class'}
    for name, image in raster_exports.items():
        export_tasks.append(start_image_export(image, name, ANALYSIS_SCALE_M, 255 if name in categorical else -9999))

    study_boundaries = summary_zones.merge(reaches)
    export_tasks.extend([
        start_table_export(study_boundaries, 'V00_study_and_reach_boundaries', 'GeoJSON'),
        start_table_export(waypoint_fc.merge(reference_point_fc), 'V01_waypoints_and_reference_points', 'GeoJSON'),
        start_table_export(pre_class_area_fc, 'T01_pre_class_area_by_zone', 'CSV'),
        start_table_export(post_class_area_fc, 'T02_post_class_area_by_zone', 'CSV'),
        start_table_export(change_area_fc, 'T03_change_area_by_zone', 'CSV'),
        start_table_export(transition_area_fc, 'T04_pre_post_transition_area', 'CSV'),
        start_table_export(sensitivity_fc, 'T05_threshold_sensitivity', 'CSV'),
    ])
    if source_vector_count:
        export_tasks.append(start_table_export(source_candidate_vectors, 'V02_source_disturbance_candidates', 'GeoJSON'))
    if temporary_water_count:
        export_tasks.append(start_table_export(temporary_water_vectors, 'V03_temporary_water_candidates', 'GeoJSON'))
    if debris_vector_count:
        export_tasks.append(start_table_export(debris_vectors, 'V04_debris_mudflow_candidates', 'GeoJSON'))
    if disturbance_vector_count:
        export_tasks.append(start_table_export(disturbance_vectors, 'V05_disturbed_terrain_candidates', 'GeoJSON'))

    task_rows = []
    for task in export_tasks:
        status = task.status()
        task_rows.append({'id': status.get('id'), 'description': status.get('description'), 'state': status.get('state')})
    export_task_df = pd.DataFrame(task_rows)
    display(export_task_df)
else:
    print('Drive exports are disabled. Set START_DRIVE_EXPORTS = True and rerun this cell when maps are acceptable.')

# %% [markdown] cell 47
# ## 19. Save a local audit bundle
#
# This writes small tables, threshold settings, dataset identifiers, scene IDs, and a methods note. Earth Engine raster/vector exports remain separate Drive tasks.

# %% cell 48
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

frames = {
    's2_pre_scene_inventory.csv': s2_pre_inventory_df,
    's2_post_scene_inventory.csv': s2_post_inventory_df,
    's1_scene_inventory.csv': s1_inventory_df,
    's1_track_candidates.csv': s1_track_candidates_df,
    'landsat9_scene_inventory.csv': landsat_inventory_df,
    'optical_coverage.csv': coverage_df,
    'pre_class_area_by_zone.csv': pre_class_area_df,
    'post_class_area_by_zone.csv': post_class_area_df,
    'class_area_pre_post_comparison.csv': class_area_comparison_df,
    'change_area_by_zone.csv': change_area_df,
    'pre_post_transition_area.csv': transition_area_df,
    'threshold_sensitivity.csv': sensitivity_df,
    'source_candidate_attributes.csv': source_candidate_df,
    'temporary_water_candidate_attributes.csv': temporary_water_df,
    'rf_validation.csv': rf_validation_df,
    'rf_feature_importance.csv': rf_feature_importance_df,
    'high_resolution_inventory.csv': high_res_inventory_df,
    'cems_comparison.csv': cems_comparison_df,
}
for filename, frame in frames.items():
    if isinstance(frame, pd.DataFrame):
        frame.to_csv(LOCAL_OUTPUT_DIR / filename, index=False)

manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'event_time_utc': EVENT_TIME_UTC,
    'earth_engine_project': EE_PROJECT,
    'dataset_ids': {
        'sentinel2_sr': S2_COLLECTION_ID,
        'sentinel2_cloud_probability': S2_CLOUD_COLLECTION_ID,
        'sentinel1': s1_collection_id,
        'landsat9': L9_COLLECTION_ID,
        'copernicus_dem': dem_collection_id,
        'merit_hydro': 'MERIT/Hydro/v1_0_1',
        'jrc_surface_water': 'JRC/GSW1_4/GlobalSurfaceWater',
        'worldcover': 'ESA/WorldCover/v200',
    },
    'date_windows': {
        's2_primary_pre_start': S2_PRIMARY_PRE_START,
        's2_fallback_pre_start': S2_FALLBACK_PRE_START,
        's2_fallback_pre_end': S2_FALLBACK_PRE_END,
        's2_pre_end': S2_PRE_END,
        's2_post_start': S2_POST_START,
        's2_post_end': S2_POST_END,
        's1_pre_start': S1_PRE_START,
        's1_post_end': S1_POST_END,
    },
    'geometry': {
        'source_search_lon_lat': SOURCE_SEARCH_LON_LAT,
        'usgs_signal_lon_lat': USGS_SIGNAL_LON_LAT,
        'source_radius_m': SOURCE_SEARCH_RADIUS_M,
        'corridor_half_width_m': CORRIDOR_HALF_WIDTH_M,
        'approximate_centreline_km': cumulative_km,
        'waypoints': waypoints,
    },
    'processing': {
        'analysis_scale_m': ANALYSIS_SCALE_M,
        'output_crs': OUTPUT_CRS,
        's2_pre_mask_method': s2_pre_mask_method,
        's2_post_mask_method': s2_post_mask_method,
        's1_available': S1_AVAILABLE,
        's1_collection': s1_collection_id,
        's1_pass': selected_s1_pass,
        's1_relative_orbit': selected_s1_orbit,
        'landsat_available': LANDSAT_AVAILABLE,
    },
    'thresholds': {
        'ndsi_snow': NDSI_SNOW_THRESHOLD,
        'mndwi_water': MNDWI_WATER_THRESHOLD,
        'ndvi_vegetation': NDVI_VEGETATION_THRESHOLD,
        'spectral_rms': SPECTRAL_RMS_THRESHOLD,
        'spectral_angle_rad': SPECTRAL_ANGLE_THRESHOLD_RAD,
        'dnbr_abs': DNBR_ABS_THRESHOLD,
        'sar_change_db': SAR_CHANGE_DB_THRESHOLD,
        'minimum_connected_pixels': MIN_CONNECTED_PIXELS,
    },
    'class_names': CLASS_NAMES,
    'materialised_result_assets': RESULT_ASSET_IDS,
    'materialised_result_band_order': RESULT_BAND_ORDER,
    'materialised_configuration_sha256': RESULT_CONFIG_SHA256,
    'materialised_class_nodata': RESULT_CLASS_NODATA,
    'optional_assets': {
        'training_asset': CUSTOM_TRAINING_ASSET,
        'cems_reference_asset': CEMS_REFERENCE_ASSET,
        'high_resolution_collections': OPTIONAL_EVENT_COLLECTIONS,
    },
}
(LOCAL_OUTPUT_DIR / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
(LOCAL_OUTPUT_DIR / 'class_legend.csv').write_text('class_id,class_name,hex_color\n' + '\n'.join(
    f'{i},"{CLASS_NAMES[i]}",#{CLASS_PALETTE[i]}' for i in sorted(CLASS_NAMES)
))

methods_note = (
    'Pre-event Sentinel-2 surface-reflectance scenes from 22–26 August 2026 were prioritised, '
    'with 10–16 August scenes used as clear-pixel fallback. Post-event Sentinel-2 scenes from '
    '26 August–1 September 2026 were mosaicked by closest valid pixel. Cloud probability was '
    'used when matched and SCL-only masking otherwise. Optical change was restricted to pixels '
    'valid before and after the event and above a minimum terrain-illumination threshold. '
    'Sentinel-1 used the closest pass/relative-orbit combination with observations on both sides '
    'of the event, median compositing and 30 m focal-median speckle suppression. Candidate water, '
    'snow/ice loss, debris and disturbance masks combined explicit spectral, SAR, terrain, drainage, '
    'historical-water and connected-component rules. Results are provisional pending high-resolution '
    'interpretation and independent validation.'
)
(LOCAL_OUTPUT_DIR / 'methods_note.txt').write_text(methods_note)

zip_path = LOCAL_OUTPUT_DIR.parent / 'Lhende_2026_local_audit_bundle.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(LOCAL_OUTPUT_DIR.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(LOCAL_OUTPUT_DIR.parent))
print('Local audit bundle:', zip_path)

if MOUNT_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    drive_target = Path('/content/drive/MyDrive/Lhende_2026_GEE_Audit')
    drive_target.mkdir(parents=True, exist_ok=True)
    for path in LOCAL_OUTPUT_DIR.iterdir():
        if path.is_file():
            (drive_target / path.name).write_bytes(path.read_bytes())
    print('Audit files copied to:', drive_target)

if DOWNLOAD_LOCAL_AUDIT_ZIP:
    from google.colab import files
    files.download(str(zip_path))

# %% [markdown] cell 49
# ## 20. QGIS assembly and validation
#
# 1. Run all analytical cells with exports initially disabled; inspect cloud/shadow validity and source candidates.
# 2. Adjust thresholds only with documented high-resolution evidence; rerun the sensitivity table after every change.
# 3. Enable Drive exports and download the completed GeoTIFF/GeoJSON/CSV outputs.
# 4. In QGIS, use project CRS **EPSG:32645**, apply the class legend, and keep validity masks visible during interpretation.
# 5. Clip EMSR927 observed-event layers to matching spatial support before calculating overlap.
# 6. Create independent stratified validation polygons across all seven classes and unchanged controls; never sample them from the rule map.
# 7. Report confusion matrix, per-class producer/user accuracy, overall accuracy, kappa, mapped-area-adjusted estimates where feasible, and threshold sensitivity.
# 8. Treat WorldCover built-up pixels as exposure context only; claim infrastructure damage only where corroborated by CEMS or high-resolution imagery.
# 9. Manually inspect all candidate temporary lakes/dams for cloud, terrain shadow, layover, and no-data artefacts.
# 10. Archive the notebook, audit bundle, Earth Engine task metadata, exported assets, QGIS project, and source licences.
#
# ### Core data-source identifiers
#
# - `COPERNICUS/S2_SR_HARMONIZED`
# - `COPERNICUS/S2_CLOUD_PROBABILITY`
# - `COPERNICUS/S1_GRD` / `COPERNICUS/S1_GRD_FLOAT`
# - `LANDSAT/LC09/C02/T1_L2`
# - `COPERNICUS/DEM/GLO30_2024_1` (legacy fallback: `COPERNICUS/DEM/GLO30`)
# - `MERIT/Hydro/v1_0_1`
# - `JRC/GSW1_4/GlobalSurfaceWater`
# - `ESA/WorldCover/v200`
# - optional Vantor/Planet event collections listed above
# - optional Copernicus EMS Rapid Mapping activation EMSR927 reference vectors
