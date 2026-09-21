from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform, unary_union
from pyproj import CRS, Transformer


def inventory_patches(m01_path, zones_geojson, output_dir, zone_field="zone_id", connectivity=4):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if connectivity not in (4,8): raise ValueError("connectivity must be 4 or 8")
    with rasterio.open(m01_path) as ds:
        m=ds.read(1); observed=m!=ds.nodata if ds.nodata is not None else np.ones(m.shape,bool)
        cand=observed & (m==1)
        gj=json.loads(Path(zones_geojson).read_text(encoding="utf-8")); feats=gj.get("features",[])
        if not feats: raise ValueError("Zones GeoJSON has no features")
        zcrs=CRS.from_user_input(gj.get("crs",{}).get("properties",{}).get("name") or "EPSG:4326")
        rcrs=CRS.from_user_input(ds.crs); tf=Transformer.from_crs(zcrs,rcrs,always_xy=True)
        zone_shapes=[]; zone_names=[]
        for i,f in enumerate(feats,1):
            zid=str(f.get("properties",{}).get(zone_field,f"zone_{i}")); g=shape(f["geometry"])
            if zcrs!=rcrs: g=shp_transform(tf.transform,g)
            zone_shapes.append((g,i)); zone_names.append((i,zid))
        zones=rasterize(zone_shapes,out_shape=m.shape,transform=ds.transform,fill=0,all_touched=False,dtype="int32")
        structure=ndi.generate_binary_structure(2,1 if connectivity==4 else 2)
        id_raster=np.zeros(m.shape,np.int32); id_raster[~observed]=-9999
        records=[]; gid=0; px_area=abs(ds.transform.a*ds.transform.e)
        to_wgs=Transformer.from_crs(rcrs,4326,always_xy=True)
        for zcode,zid in zone_names:
            labeled,n=ndi.label(cand & (zones==zcode),structure=structure)
            comps=[]
            for lid in range(1,n+1):
                idx=np.argwhere(labeled==lid); comps.append((len(idx),lid,idx))
            comps.sort(key=lambda x:(-x[0], int(x[2][:,0].min()), int(x[2][:,1].min())))
            for rank,(count,lid,idx) in enumerate(comps,1):
                gid+=1; id_raster[labeled==lid]=gid
                geoms=[]
                for geom,val in shapes((labeled==lid).astype("uint8"),mask=(labeled==lid),transform=ds.transform):
                    if val==1: geoms.append(shape(geom))
                geom=unary_union(geoms)
                c=geom.centroid; lon,lat=to_wgs.transform(c.x,c.y)
                records.append({"patch_id":f"{zid}_P{rank:04d}","raster_id":gid,"zone_id":zid,
                    "area_rank_in_zone":rank,"pixel_count":count,"area_km2":count*px_area/1e6,
                    "review_lon":lon,"review_lat":lat,"geometry":geom})
        prof=ds.profile.copy(); prof.update(dtype="int32",count=1,nodata=-9999,compress="deflate")
        with rasterio.open(output_dir/"patch_ids.tif","w",**prof) as out: out.write(id_raster,1)
        props=[]; features=[]
        for r in records:
            q={k:v for k,v in r.items() if k!="geometry"}; props.append(q)
            features.append({"type":"Feature","properties":q,"geometry":mapping(r["geometry"])})
        pd.DataFrame(props).to_csv(output_dir/"patch_inventory.csv",index=False)
        crs_name=str(ds.crs)
        geo={"type":"FeatureCollection","name":"candidate_patches","crs":{"type":"name","properties":{"name":crs_name}},"features":features}
        (output_dir/"candidate_patches.geojson").write_text(json.dumps(geo),encoding="utf-8")
        summary=pd.DataFrame(props).groupby("zone_id",as_index=False).agg(patch_count=("patch_id","count"),candidate_area_km2=("area_km2","sum"),largest_patch_km2=("area_km2","max")) if props else pd.DataFrame()
        summary.to_csv(output_dir/"patch_summary.csv",index=False)
        return pd.DataFrame(props), summary
