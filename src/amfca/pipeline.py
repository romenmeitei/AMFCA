from __future__ import annotations
from pathlib import Path
import json, runpy
from .config import load_config, validate_config
from .provenance import write_json, environment_record
from .legacy import load_stage, frozen_dir
from .core_render import render_assignments
from .concurrence import quantify_concurrence
from .patches import inventory_patches

LHENDE_ORDER=["m01_concurrence","background_comparison","patch_inventory","s2_recovery","directional","dated_scene_audit","inventory_reconciliation","top20_review","external_corroboration"]


def plan(cfg):
    profile=cfg.get("project",{}).get("profile","lhende_exact")
    if profile=="generic": return ["generic_concurrence","generic_patches"]
    return LHENDE_ORDER.copy()


def _path(v): return str(Path(v).expanduser()) if v else ""


def run_generic(cfg, repo_root="."):
    out=Path(cfg["outputs"]["run_root"]); out.mkdir(parents=True,exist_ok=True)
    i=cfg["inputs"]; a=cfg["analysis"]
    cdir=out/"01_concurrence"
    table=quantify_concurrence(i["sar_control_pattern"],i["spectral_rms"],i["zones"],cdir,
        threshold=float(a["primary_rms_threshold"]),sensitivity=a["rms_thresholds"],zone_field=a.get("zone_field","zone_id"))
    pdir=out/"02_patches"
    inv,summary=inventory_patches(cdir/"M01_concurrence.tif",i["zones"],pdir,zone_field=a.get("zone_field","zone_id"),connectivity=int(a.get("connectivity",4)))
    write_json(out/"run_manifest.json",{"profile":"generic","environment":environment_record(repo_root),"config":cfg})
    return {"run_root":str(out),"concurrence":table,"patch_inventory":inv,"patch_summary":summary}


def run_lhende(cfg, repo_root=".", stages=None):
    stages=stages or LHENDE_ORDER
    unknown=set(stages)-set(LHENDE_ORDER)
    if unknown: raise ValueError(f"Unknown Lhende stage(s): {sorted(unknown)}")
    p=cfg["project"]; i=cfg["inputs"]; o=cfg["outputs"]; a=cfg["analysis"]
    state={}

    if "m01_concurrence" in stages:
        mod,src=load_stage("m01_concurrence",repo_root)
        files={"sar":_path(i["sar_control_pattern"]),"optical":_path(i["spectral_rms"]),"zones":_path(i["zones"])}
        if i.get("qgis_m01"): files["qgis_m01"]=_path(i["qgis_m01"])
        mod.PRIMARY_THRESHOLD=float(a["primary_rms_threshold"]); mod.RMS_THRESHOLDS=list(a["rms_thresholds"])
        mod.run_m01_quantification(files, bool(o.get("save_to_drive",True)), str(o.get("local_concurrence_root","/content/Lhende_2026_concurrence")))
        state["m01_dir"]=str(mod.M01_DRIVE_DIR or mod.M01_OUTPUT_DIR)

    m01=state.get("m01_dir") or i.get("existing_m01_audit_dir")
    if "background_comparison" in stages:
        if not m01: raise ValueError("background_comparison requires m01 audit dir")
        mod,src=load_stage("background_comparison",repo_root)
        mod.run_background_comparison(base_audit=Path(m01),save_to_drive=bool(o.get("save_to_drive",True)),output_root=str(o.get("local_background_root","/content/Lhende_2026_M01_Background")))
        state["background_dir"]=str(getattr(mod,"M01_BACKGROUND_DRIVE_DIR",None) or getattr(mod,"M01_BACKGROUND_OUTPUT_DIR",None))

    if "patch_inventory" in stages:
        if not m01: raise ValueError("patch_inventory requires m01 audit dir")
        mod,src=load_stage("patch_inventory",repo_root)
        r=mod.run_candidate_patch_inventory(audit_dir=Path(m01),save_to_drive=bool(o.get("save_to_drive",True)),
            local_output_root=str(o.get("local_patch_root","/content/Lhende_2026_Candidate_Patches")),
            drive_output_root=str(o.get("drive_patch_root","/content/drive/MyDrive/Lhende_2026_Candidate_Patches")),
            review_n=int(a.get("review_n",10)),source_code_path=src,verbose=True)
        state["patch_dir"]=str(r["drive_dir"] or r["local_dir"])

    patch=state.get("patch_dir") or i.get("existing_patch_dir")
    if "s2_recovery" in stages:
        if not m01: raise ValueError("s2_recovery requires m01 audit dir")
        mod,src=load_stage("s2_recovery",repo_root); c=dict(mod.CONFIG)
        c.update({"audit_dir":str(m01),"export_dir":_path(i["gee_export_dir"]),"save_to_drive":bool(o.get("save_to_drive",True)),
                  "drive_output_root":str(o.get("drive_s2_recovery_root","/content/drive/MyDrive/Lhende_2026_S2_Review"))})
        r=mod.run_s2_recovery(c,source_code_path=src); state["s2_recovery_dir"]=str(r["drive_dir"] or r["local_dir"])

    s2=state.get("s2_recovery_dir") or i.get("existing_s2_recovery_dir")
    if "directional" in stages:
        if not (patch and s2): raise ValueError("directional requires patch_dir and s2_recovery_dir")
        mod,src=load_stage("directional",repo_root); c=dict(mod.PATCH_DIRECTIONAL_CONFIG)
        c.update({"patch_id":a.get("directional_patch_id","R05_P0001"),"patch_dir":str(patch),"s2_dir":str(s2),"export_dir":_path(i["gee_export_dir"]),"save_to_drive":bool(o.get("save_to_drive",True))})
        r=mod.run_patch_directional(c,source_code_path=src); state["directional_dir"]=str(r["drive_dir"] or r["local_dir"])

    if "dated_scene_audit" in stages:
        if not (patch and s2): raise ValueError("dated_scene_audit requires patch_dir and s2_recovery_dir")
        mod,src=load_stage("dated_scene_audit",repo_root); c=dict(mod.SCENE_AUDIT_CONFIG)
        c.update({"ee_project":p["ee_project"],"patch_id":a.get("directional_patch_id","R05_P0001"),"patch_dir":str(patch),"s2_recovery_dir":str(s2),
                  "original_audit_dir":_path(i["original_gee_audit_dir"]),"event_time_utc":p["event_time_utc"],"save_to_drive":bool(o.get("save_to_drive",True))})
        if a.get("dated_scene_review_end_exclusive_utc") is not None: c["review_end_exclusive_utc"]=a.get("dated_scene_review_end_exclusive_utc")
        r=mod.run_scene_audit(c,source_code_path=src); state["dated_scene_audit_dir"]=str(r["drive_dir"] or r["local_dir"])

    dated=state.get("dated_scene_audit_dir") or i.get("existing_dated_scene_audit_dir")
    if "inventory_reconciliation" in stages:
        if not dated: raise ValueError("inventory_reconciliation requires dated_scene_audit_dir")
        mod,src=load_stage("inventory_reconciliation",repo_root); c=dict(mod.CONFIG)
        c.update({"audit_dir":str(dated),"save_to_drive":bool(o.get("save_to_drive",True))})
        r=mod.run_inventory_reconciliation(c,source_code_path=src); state["inventory_reconciliation_dir"]=str(r["drive_dir"] or r["local_dir"])

    if "top20_review" in stages:
        if not (patch and s2): raise ValueError("top20_review requires patch_dir and s2_recovery_dir")
        mod,src=load_stage("top20_review",repo_root); c=dict(mod.CONFIG)
        c.update({"patch_dir":str(patch),"s2_recovery_dir":str(s2),"original_audit_dir":_path(i["original_gee_audit_dir"]),"ee_project":p["ee_project"],
                  "top_n_per_reach":int(a.get("top_n_per_reach",10)),"output_root":str(o.get("drive_top20_root","/content/drive/MyDrive/Lhende_2026_Top20_Patch_Review"))})
        r=mod.run_batch_review(c,source_code_path=src); state["top20_dir"]=str(r["output_dir"])

    top20=state.get("top20_dir") or i.get("existing_top20_dir")
    if "external_corroboration" in stages:
        if not top20: raise ValueError("external_corroboration requires top20_dir")
        mod,src=load_stage("external_corroboration",repo_root); c=dict(mod.CONFIG)
        c.update({"top20_run_dir":str(top20),"ee_project":p["ee_project"],"output_root":str(o.get("drive_external_root","/content/drive/MyDrive/Lhende_2026_External_Corroboration"))})
        r=mod.run_external_corroboration(c,source_code_path=src); state["external_dir"]=str(r["output_dir"])

    write_json(Path(o.get("state_file","/content/amfca_lhende_pipeline_state.json")),{"environment":environment_record(repo_root),"state":state,"config":cfg})
    return state


def run_config(config_path, repo_root=".", stages=None):
    cfg=load_config(config_path); errors=validate_config(cfg)
    if errors: raise ValueError("Invalid configuration: " + "; ".join(errors))
    if cfg["project"].get("profile","lhende_exact")=="generic": return run_generic(cfg,repo_root)
    return run_lhende(cfg,repo_root,stages=stages)
