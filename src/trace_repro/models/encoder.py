from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def _unwrap_state_dict(checkpoint):
    state = checkpoint
    for key in ("state_dict", "model", "encoder", "backbone"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    cleaned = {}
    for key, value in state.items():
        name = key
        prefixes = ("module.", "model.", "encoder.", "backbone.", "resnet.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]; changed = True
        if name.startswith("stem.0."): name = "conv1." + name[len("stem.0."):]
        elif name.startswith("stem.1."): name = "bn1." + name[len("stem.1."):]
        cleaned[name] = value
    return cleaned


def _build_resnet50(pretrained=True, checkpoint=None):
    from torchvision.models import resnet50
    # torchvision 0.12 in the target environment uses `pretrained`, whereas
    # recent releases use `weights`. Construct without network access first.
    try:
        model = resnet50(weights=None)
    except TypeError:
        model = resnet50(pretrained=False)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError("Encoder checkpoint not found: {}".format(checkpoint))
        state = _unwrap_state_dict(torch.load(str(checkpoint), map_location="cpu"))
        compatible = {k: v for k, v in state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        required = ["conv1.weight", "layer1.0.conv1.weight", "layer2.0.conv1.weight", "layer3.0.conv1.weight"]
        missing_required = [k for k in required if k not in compatible]
        if missing_required:
            raise RuntimeError("Checkpoint is not a compatible ResNet-50 encoder; missing {}".format(missing_required))
        model.load_state_dict(compatible, strict=False)
    elif pretrained:
        candidates = [
            Path.home() / ".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth",
            Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth",
        ]
        checkpoint = next((p for p in candidates if p.is_file()), None)
        if checkpoint is None:
            # Let torchvision use its normal cache/download behavior only as a fallback.
            try:
                return resnet50(weights="IMAGENET1K_V2")
            except (TypeError, ValueError):
                return resnet50(pretrained=True)
        state = torch.load(str(checkpoint), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = {"fc.weight", "fc.bias"}
        if set(missing) - allowed or unexpected:
            raise RuntimeError("Incompatible cached ResNet-50 state: missing={} unexpected={}".format(missing, unexpected))
    return model


class SharedEncoder(nn.Module):
    """Single ResNet-50 truncated after layer3 (1024 x 24 x 24 at 384 px)."""

    def __init__(self, pretrained=True, checkpoint=None):
        super().__init__()
        base = _build_resnet50(pretrained=pretrained, checkpoint=checkpoint)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.out_channels = 1024

    def forward(self, images):
        feature_map = self.layer3(self.layer2(self.layer1(self.stem(images))))
        global_descriptor = F.normalize(feature_map.mean(dim=(-2, -1)), dim=-1)
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = F.normalize(tokens, dim=-1)
        return feature_map, global_descriptor, tokens

