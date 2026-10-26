from pathlib import Path
import copy
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = path.parent.parent
    for section, key in (("data", "root"), ("project", "output_dir"),
                         ("data", "biomarker_csv")):
        if cfg[section].get(key) is None:
            continue
        value = Path(cfg[section][key])
        if not value.is_absolute():
            cfg[section][key] = str((base / value).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    included = list(cfg["data"]["included_classes"])
    excluded = set(cfg["data"].get("excluded_classes", []))
    if len(included) != 6 or len(set(included)) != 6:
        raise ValueError("This task requires exactly six unique included classes")
    if "Others" not in excluded or "Others" in included:
        raise ValueError("Others must be explicitly excluded from the six-class task")
    if cfg["project"].get("mode") != "full":
        raise ValueError("Paper.pdf requires project.mode=full")
    if int(cfg["data"].get("num_folds", 0)) != 5:
        raise ValueError("Paper.pdf requires five-fold cross-validation")
    expected = {"biomarker_prototypes_per_class": 8, "pathology_prototypes_per_class": 10}
    for key, value in expected.items():
        if int(cfg["model"][key]) != value: raise ValueError("{} must be {}".format(key, value))
    if abs(float(cfg["train"]["lambda_bio"]) - 0.2) > 1e-12:
        raise ValueError("lambda_bio must be 0.2")
    if abs(float(cfg["model"]["match_epsilon"]) - 1e-6) > 1e-12:
        raise ValueError("match_epsilon must be 1e-6")
    return cfg


def with_overrides(cfg, **kwargs):
    out = copy.deepcopy(cfg)
    for dotted, value in kwargs.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    validate_config(out)
    return out

