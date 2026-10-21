#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trace_repro.config import load_config
from trace_repro.training import run_experiment
from trace_repro.utils import device_from_arg


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(ROOT/"configs/trace.yaml"))
    p.add_argument("--experiment",choices=["fold1","fold2","fold3","fold4","fold5"],required=True); p.add_argument("--device",default="auto")
    args=p.parse_args(); cfg=load_config(args.config); device=device_from_arg(args.device)
    print("device={}".format(device),flush=True); result=run_experiment(cfg,args.experiment,device)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()

