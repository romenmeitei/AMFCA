from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from amfca.concurrence import quantify_concurrence
from amfca.patches import inventory_patches

def _write(path,arr,nodata,dtype):
    with rasterio.open(path,"w",driver="GTiff",height=arr.shape[0],width=arr.shape[1],count=1,dtype=dtype,crs="EPSG:32645",transform=from_origin(500000,3100000,20,20),nodata=nodata) as d:
        d.write(arr.astype(dtype),1)

def test_concurrence_and_patches(tmp_path):
    sar=np.array([[0,3,3,0],[0,3,0,0],[0,0,3,3],[0,0,3,0]],dtype=np.uint8)
    rms=np.array([[.01,.09,.07,.02],[.01,.11,.01,.01],[.01,.01,.12,.13],[.01,.01,.09,.01]],dtype=np.float32)
    _write(tmp_path/"sar.tif",sar,255,"uint8"); _write(tmp_path/"rms.tif",rms,-9999,"float32")
    # whole raster split vertically into Z1/Z2 in raster CRS
    zones={"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:32645"}},"features":[
      {"type":"Feature","properties":{"zone_id":"Z1"},"geometry":{"type":"Polygon","coordinates":[[[500000,3099920],[500040,3099920],[500040,3100000],[500000,3100000],[500000,3099920]]]}},
      {"type":"Feature","properties":{"zone_id":"Z2"},"geometry":{"type":"Polygon","coordinates":[[[500040,3099920],[500080,3099920],[500080,3100000],[500040,3100000],[500040,3099920]]]}}
    ]}
    (tmp_path/"zones.geojson").write_text(json.dumps(zones))
    tab=quantify_concurrence(tmp_path/"sar.tif",tmp_path/"rms.tif",tmp_path/"zones.geojson",tmp_path/"c",.08,[.06,.08,.10])
    assert (tmp_path/"c/M01_concurrence.tif").is_file()
    primary=tab[np.isclose(tab.rms_threshold,.08)]
    assert set(primary.zone_id)=={"Z1","Z2"}
    inv,summary=inventory_patches(tmp_path/"c/M01_concurrence.tif",tmp_path/"zones.geojson",tmp_path/"p")
    assert len(inv)>=2
    assert inv.pixel_count.sum()==5
    assert (tmp_path/"p/candidate_patches.geojson").is_file()

def test_grid_mismatch_is_refused(tmp_path):
    a=np.zeros((2,2),dtype=np.uint8); b=np.zeros((3,2),dtype=np.float32)
    _write(tmp_path/"a.tif",a,255,"uint8"); _write(tmp_path/"b.tif",b,-9999,"float32")
    zones={"type":"FeatureCollection","features":[{"type":"Feature","properties":{"zone_id":"Z"},"geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}}]}
    (tmp_path/"z.geojson").write_text(json.dumps(zones))
    try: quantify_concurrence(tmp_path/"a.tif",tmp_path/"b.tif",tmp_path/"z.geojson",tmp_path/"o")
    except ValueError as e: assert "same grid" in str(e)
    else: raise AssertionError("Expected exact-grid refusal")
