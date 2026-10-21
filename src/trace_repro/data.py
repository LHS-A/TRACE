import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset

from .utils import atomic_json_dump, ensure_dir, read_csv, write_csv


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BIOMARKER_COLUMNS = ("CD", "AVG", "SD", "CV", "MAX", "6A")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def discover_images(root, included_classes, excluded_classes=("Others",)):
    root = Path(root).resolve()
    included = list(included_classes)
    excluded = set(excluded_classes)
    if "Others" not in excluded:
        raise ValueError("Others must remain excluded")
    mapping = {name: idx for idx, name in enumerate(sorted(included))}
    records = []
    for class_name in mapping:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError("Missing class directory: {}".format(class_dir))
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(root)),
                    "class_name": class_name,
                    "label": mapping[class_name],
                    # Files sharing the leading acquisition timestamp are
                    # same-session/source derivatives and must stay together.
                    "group_id": "{}:{}".format(class_name, path.stem.split("_")[0]),
                })
    if any(r["class_name"] in excluded for r in records):
        raise AssertionError("Excluded class entered discovered records")
    return records, mapping


def _sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def audit_images(records, compute_hashes=False):
    valid, corrupt = [], []
    hashes = defaultdict(list)
    sizes, modes = Counter(), Counter()
    for rec in records:
        try:
            with Image.open(rec["path"]) as im:
                im.verify()
            with Image.open(rec["path"]) as im:
                sizes[str(im.size)] += 1
                modes[im.mode] += 1
            valid.append(rec)
            if compute_hashes:
                hashes[_sha256(rec["path"])].append(rec)
        except Exception as exc:
            corrupt.append({"path": rec["path"], "error": repr(exc)})
    duplicate_groups = []
    cross_class_duplicates = []
    if compute_hashes:
        for digest, group in hashes.items():
            if len(group) > 1:
                item = {"sha256": digest, "paths": [x["path"] for x in group],
                        "classes": sorted(set(x["class_name"] for x in group))}
                duplicate_groups.append(item)
                if len(item["classes"]) > 1:
                    cross_class_duplicates.append(item)
    report = {
        "total_discovered": len(records), "valid": len(valid), "corrupt": corrupt,
        "class_counts": dict(Counter(x["class_name"] for x in valid)),
        "sizes": dict(sizes), "modes": dict(modes),
        "hashes_computed": bool(compute_hashes),
        "duplicate_groups": duplicate_groups,
        "cross_class_duplicates": cross_class_duplicates,
        "patient_id_status": "unavailable: no documented reliable patient identifier",
        "leakage_control": "acquisition/source-level grouping by leading timestamp in filename",
    }
    return valid, report


def attach_biomarkers(records, csv_path, dataset_root, patient_id_column="patient_id"):
    """Join frozen ProtoCEC-derived training biomarkers to image records.

    No measurements are inferred here: full TRACE requires an auditable offline
    table produced from the frozen segmentation model.
    """
    if not csv_path:
        raise FileNotFoundError("Full TRACE requires data.biomarker_csv with offline ProtoCEC measurements")
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError("Biomarker table not found: {}".format(path))
    aliases = {
        "CD": ("CD", "ECD", "cell_density", "endothelial_cell_density"),
        "AVG": ("AVG", "MCA", "mean_cell_area"),
        "SD": ("SD", "S.D.", "cell_area_sd"),
        "CV": ("CV", "C.V.", "coefficient_of_variation"),
        "MAX": ("MAX", "maximum_cell_area"),
        "6A": ("6A", "HEX", "hexagonal_percentage"),
    }
    table = read_csv(path)
    if not table:
        raise ValueError("Biomarker table is empty: {}".format(path))
    keys = set(table[0])
    path_column = "path" if "path" in keys else "relative_path" if "relative_path" in keys else None
    if path_column is None or patient_id_column not in keys:
        raise ValueError("Biomarker CSV needs path/relative_path and {}".format(patient_id_column))
    resolved = {}
    for canonical, choices in aliases.items():
        source = next((name for name in choices if name in keys), None)
        if source is None:
            raise ValueError("Missing biomarker {} (accepted aliases: {})".format(canonical, choices))
        resolved[canonical] = source
    root = Path(dataset_root).resolve()
    by_path = {}
    for row in table:
        source = Path(row[path_column])
        source = source.resolve() if source.is_absolute() else (root / source).resolve()
        if source in by_path:
            raise ValueError("Duplicate biomarker row for {}".format(source))
        by_path[source] = row
    merged, missing = [], []
    for record in records:
        key = Path(record["path"]).resolve()
        row = by_path.get(key)
        if row is None:
            missing.append(str(key)); continue
        item = dict(record)
        item["patient_id"] = str(row[patient_id_column]).strip()
        if not item["patient_id"]:
            raise ValueError("Empty patient ID for {}".format(key))
        for canonical, source in resolved.items():
            item[canonical] = float(row[source])
            if not np.isfinite(item[canonical]):
                raise ValueError("Non-finite {} for {}".format(canonical, key))
        item["group_id"] = item["patient_id"]
        merged.append(item)
    if missing:
        raise ValueError("Biomarkers missing for {} images; first: {}".format(len(missing), missing[:5]))
    return merged


def create_stratified_five_folds(records, seed=42, n_splits=5):
    """Patient-disjoint stratified CV with one validation and one held-out fold."""
    if n_splits != 5:
        raise ValueError("Paper protocol requires exactly five folds")
    labels = np.asarray([int(r["label"]) for r in records])
    groups = np.asarray([r.get("patient_id", r.get("group_id")) for r in records])
    if any(x in (None, "") for x in groups):
        raise ValueError("Patient identifiers are required for patient-disjoint five-fold CV")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    held_out = [test for _, test in splitter.split(np.zeros(len(records)), labels, groups)]
    folds = {}
    all_indices = set(range(len(records)))
    for fold in range(n_splits):
        test_idx = set(held_out[fold].tolist())
        val_idx = set(held_out[(fold + 1) % n_splits].tolist())
        train_idx = all_indices - test_idx - val_idx
        split = {
            "train": [dict(records[i], split="train") for i in sorted(train_idx)],
            "val": [dict(records[i], split="val") for i in sorted(val_idx)],
            "test": [dict(records[i], split="test") for i in sorted(test_idx)],
        }
        assert_split_integrity(split)
        group_sets = {name: {r.get("patient_id", r["group_id"]) for r in rows} for name, rows in split.items()}
        if any(group_sets[a] & group_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise AssertionError("Patient leakage between CV partitions")
        folds["fold{}".format(fold + 1)] = split
    return folds


def create_stratified_splits(records, seed, ratios=(0.7, 0.1, 0.2)):
    indices = np.arange(len(records))
    labels = np.asarray([int(r["label"]) for r in records])
    train_ratio, val_ratio, test_ratio = ratios
    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(outer.split(indices, labels))
    inner_val = val_ratio / (train_ratio + val_ratio)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=inner_val, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(train_val_idx, labels[train_val_idx]))
    result = {
        "train": [dict(records[i], split="train") for i in train_val_idx[train_rel]],
        "val": [dict(records[i], split="val") for i in train_val_idx[val_rel]],
        "test": [dict(records[i], split="test") for i in test_idx],
    }
    assert_split_integrity(result)
    return result


def create_grouped_stratified_splits(records, seed, ratios=(0.7, 0.1, 0.2)):
    """Class-stratified split keeping acquisition/source groups indivisible.

    Groups are class-pure by construction. Multiple deterministic randomized
    trials select boundaries minimizing absolute count error against 70/10/20.
    """
    rng = np.random.RandomState(seed)
    result = {"train": [], "val": [], "test": []}
    labels = sorted(set(int(r["label"]) for r in records))
    for label in labels:
        groups = defaultdict(list)
        for row in records:
            if int(row["label"]) == label:
                groups[row.get("group_id", row["path"])].append(row)
        group_rows = list(groups.values()); total = sum(len(g) for g in group_rows)
        target_train, target_val = total * ratios[0], total * ratios[1]
        best = None
        # Explore orderings because indivisible groups make exact boundaries
        # a small subset-sum problem. 256 trials is ample for max group <= 18.
        for _ in range(256):
            order = rng.permutation(len(group_rows)); sizes = np.asarray([len(group_rows[i]) for i in order])
            cumulative = np.cumsum(sizes)
            a = int(np.argmin(np.abs(cumulative - target_train))) + 1
            tail = cumulative[a:] - cumulative[a - 1]
            if len(tail) == 0: continue
            b = a + int(np.argmin(np.abs(tail - target_val))) + 1
            counts = (int(cumulative[a-1]), int(cumulative[b-1]-cumulative[a-1]), int(total-cumulative[b-1]))
            error = sum(abs(counts[i] - total * ratios[i]) for i in range(3))
            if best is None or error < best[0]: best = (error, order, a, b)
        _, order, a, b = best
        for split, indices in zip(("train", "val", "test"), (order[:a], order[a:b], order[b:])):
            result[split].extend(dict(row, split=split) for idx in indices for row in group_rows[idx])
    for rows in result.values(): rng.shuffle(rows)
    assert_split_integrity(result)
    group_sets = {name: {r.get("group_id", r["path"]) for r in rows} for name, rows in result.items()}
    if group_sets["train"] & group_sets["val"] or group_sets["train"] & group_sets["test"] or group_sets["val"] & group_sets["test"]:
        raise AssertionError("Acquisition/source group leakage between splits")
    return result


def assert_split_integrity(splits):
    paths = {name: {r["path"] for r in rows} for name, rows in splits.items()}
    if paths["train"] & paths["val"] or paths["train"] & paths["test"] or paths["val"] & paths["test"]:
        raise AssertionError("Path leakage between splits")
    all_rows = sum((rows for rows in splits.values()), [])
    if len(all_rows) != len(set(r["path"] for r in all_rows)):
        raise AssertionError("Duplicate path in split manifests")
    if any(r["class_name"] == "Others" for r in all_rows):
        raise AssertionError("Others entered a split")
    for name in ("train", "test"):
        if len(set(r["label"] for r in splits[name])) != 6:
            raise AssertionError("Every class must appear in {}".format(name))


def save_splits(splits, out_dir, class_mapping, seed):
    out_dir = ensure_dir(out_dir)
    extra = ["patient_id"] + list(BIOMARKER_COLUMNS) if splits["train"] and "CD" in splits["train"][0] else []
    fields = ["path", "relative_path", "class_name", "label", "group_id", "split"] + extra
    stats = {"seed": seed, "class_mapping": class_mapping, "splits": {}}
    for name, rows in splits.items():
        write_csv(rows, out_dir / "{}.csv".format(name), fields)
        stats["splits"][name] = {
            "total": len(rows), "group_count": len(set(r.get("group_id", r["path"]) for r in rows)),
            "class_counts": dict(Counter(r["class_name"] for r in rows))
        }
    atomic_json_dump(class_mapping, out_dir / "class_mapping.json")
    atomic_json_dump(stats, out_dir / "split_stats.json")
    return stats


def load_manifest(path):
    rows = read_csv(path)
    for row in rows:
        row["label"] = int(row["label"])
        for name in BIOMARKER_COLUMNS:
            if name in row and row[name] != "": row[name] = float(row[name])
    return rows


class TraceImageDataset(Dataset):
    def __init__(self, rows, input_size=384, train=False, augmentation=None):
        self.rows = list(rows)
        self.input_size = int(input_size)
        self.train = bool(train)
        self.augmentation = augmentation or {}

    def __len__(self):
        return len(self.rows)

    def _transform(self, image):
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        if self.train:
            if torch.rand(()) < self.augmentation.get("horizontal_flip_probability", 0.5):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if torch.rand(()) < self.augmentation.get("vertical_flip_probability", 0.5):
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if self.augmentation.get("rotation_90", True):
                image = image.rotate(90 * int(torch.randint(0, 4, (1,))))
            brightness = float(self.augmentation.get("brightness", 0.1))
            contrast = float(self.augmentation.get("contrast", 0.1))
            if brightness:
                image = ImageEnhance.Brightness(image).enhance(1.0 + (2.0 * float(torch.rand(())) - 1.0) * brightness)
            if contrast:
                image = ImageEnhance.Contrast(image).enhance(1.0 + (2.0 * float(torch.rand(())) - 1.0) * contrast)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as im:
            image = self._transform(im.convert("RGB"))
        sample = {"image": image, "label": torch.tensor(row["label"], dtype=torch.long),
                  "path": row["path"], "index": torch.tensor(index)}
        if "bio_cluster" in row and row["bio_cluster"] not in ("", None):
            sample["bio_cluster"] = torch.tensor(int(row["bio_cluster"]), dtype=torch.long)
        if all(name in row for name in BIOMARKER_COLUMNS):
            sample["biomarkers"] = torch.tensor([row[name] for name in BIOMARKER_COLUMNS], dtype=torch.float32)
        return sample
