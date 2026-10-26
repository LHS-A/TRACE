#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_repro.config import load_config
from trace_repro.data import (attach_biomarkers, audit_images, create_stratified_five_folds,
                              discover_images, save_splits)
from trace_repro.utils import atomic_json_dump, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Audit six classes and create three TRACE splits")
    parser.add_argument("--config", default=str(ROOT / "configs/trace.yaml"))
    parser.add_argument("--hashes", action="store_true", help="Compute full SHA-256 duplicate audit")
    args = parser.parse_args(); cfg = load_config(args.config)
    records, mapping = discover_images(cfg["data"]["root"], cfg["data"]["included_classes"], cfg["data"]["excluded_classes"])
    records, audit = audit_images(records, compute_hashes=args.hashes)
    if cfg["project"]["mode"] != "full":
        raise ValueError("Paper.pdf specifies full dual-prototype TRACE; project.mode must be full")
    records = attach_biomarkers(records, cfg["data"].get("biomarker_csv"), cfg["data"]["root"],
                                cfg["data"].get("patient_id_column", "patient_id"))
    output = Path(cfg["project"]["output_dir"]); ensure_dir(output / "splits")
    audit["excluded_classes"] = cfg["data"]["excluded_classes"]
    audit["included_classes"] = sorted(cfg["data"]["included_classes"])
    audit["biomarker_status"] = "validated offline ProtoCEC table"
    audit["selected_mode"] = "TRACE-Full"
    atomic_json_dump(audit, output / "data_audit.json")
    print("class_mapping={}".format(mapping), flush=True)
    seed = int(cfg["data"]["cv_seed"])
    folds = create_stratified_five_folds(records, seed, int(cfg["data"]["num_folds"]))
    for experiment, splits in folds.items():
        stats = save_splits(splits, output / "splits" / experiment, mapping, seed)
        print("{}: {}".format(experiment, stats["splits"]), flush=True)


if __name__ == "__main__": main()
