#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_repro.data import IMAGENET_MEAN, IMAGENET_STD
from trace_repro.models.graph import GraphTopology
from trace_repro.models.trace import TRACEModel
from trace_repro.utils import device_from_arg


def load_image(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"]
    topology_state = checkpoint["topology"]
    topology = GraphTopology(
        num_classes=int(topology_state["num_classes"]),
        k_path=int(topology_state["k_path"]),
        k_bio=int(topology_state["k_bio"]),
        full_mode=bool(topology_state["full_mode"]),
        node_types=topology_state["node_types"],
        edge_index=topology_state["edge_index"],
        relation_names=tuple(topology_state["relation_names"]),
    )
    model = TRACEModel(
        num_classes=len(checkpoint["class_names"]),
        pathology_prototypes=state["pathology_prototypes"],
        biomarker_prototypes=state["biomarker_prototypes"],
        topology=topology,
        pretrained=False,
        eas_hidden=checkpoint["config"]["model"]["eas_hidden_dim"],
        graph_hidden=checkpoint["config"]["model"]["graph_hidden_dim"],
        graph_heads=checkpoint["config"]["model"]["graph_heads"],
        graph_layers=checkpoint["config"]["model"]["graph_layers"],
        dropout=checkpoint["config"]["model"]["dropout"],
        match_epsilon=checkpoint["config"]["model"]["match_epsilon"],
    )
    model.load_state_dict(state)
    model.set_ood_threshold(checkpoint["ood_threshold"])
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run TRACE inference on one image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    size = int(checkpoint["config"]["data"]["input_size"])
    image = load_image(args.image, size).to(device)
    with torch.inference_mode():
        output = model(image, apply_ood=True)

    prediction = int(output["prediction"][0])
    result = {
        "image": str(Path(args.image).resolve()),
        "is_ood": bool(output["is_ood"][0]),
        "coverage": float(output["coverage"][0]),
        "prediction": None if prediction < 0 else checkpoint["class_names"][prediction],
        "probabilities": {
            name: float(output["probabilities"][0, index])
            for index, name in enumerate(checkpoint["class_names"])
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
