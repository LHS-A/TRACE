import json
import math
import gc
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BIOMARKER_COLUMNS, TraceImageDataset, load_manifest
from .models.graph import build_graph_topology
from .models.trace import EncoderPretrainingModel, TRACEModel
from .prototypes import (pam_kmedoids, prototype_matching_loss, recompute_biomarker_visual_prototypes,
                         robust_standardize, weighted_spherical_kmeans, weighted_spherical_kmeans_torch)
from .utils import atomic_json_dump, ensure_dir, seed_everything, write_csv


def _loader(rows, cfg, train=False, batch_size=None):
    ds = TraceImageDataset(rows, cfg["data"]["input_size"], train=train,
                           augmentation=cfg.get("augmentation", {}))
    return DataLoader(ds, batch_size=batch_size or cfg["data"]["batch_size"], shuffle=train,
                      num_workers=cfg["data"]["num_workers"], pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg["data"]["num_workers"] > 0)


def _amp_settings(device):
    if device.type != "cuda": return False, torch.float32
    bf16 = torch.cuda.is_bf16_supported()
    return True, torch.bfloat16 if bf16 else torch.float16


def _autocast(device, enabled, dtype):
    return torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def _scheduler(optimizer, total_epochs, warmup, min_lr):
    def factor(epoch):
        if epoch < warmup: return float(epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        base = 0.5 * (1.0 + math.cos(math.pi * progress))
        max_lr = max(g["lr"] for g in optimizer.param_groups)
        return max(min_lr / max_lr, base)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def _encoder_validation_loss(model, loader, device):
    """Validation used only for checkpoint selection during training."""
    model.eval(); losses = []
    amp, dtype = _amp_settings(device)
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        with _autocast(device, amp, dtype):
            out = model(images); loss = model.classification_loss(out, labels)
        losses.append(float(loss))
    return float(np.mean(losses))


def train_encoder(train_rows, val_rows, cfg, device, out_dir, class_names):
    model = EncoderPretrainingModel(
        len(class_names), pretrained=cfg["model"]["pretrained"],
        encoder_checkpoint=cfg["model"].get("protocec_checkpoint"),
        eas_hidden=cfg["model"]["eas_hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg["train"]["backbone_lr"]},
        {"params": model.eas.parameters(), "lr": cfg["train"]["backbone_lr"]},
        {"params": list(model.global_head.parameters()) + list(model.attention_head.parameters()),
         "lr": cfg["train"]["head_lr"]}], weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["encoder_epochs"]
    scheduler = _scheduler(optimizer, epochs, cfg["train"]["warmup_epochs"], cfg["train"]["min_lr"])
    train_loader, val_loader = _loader(train_rows, cfg, True), _loader(val_rows, cfg)
    amp, dtype = _amp_settings(device); scaler = torch.cuda.amp.GradScaler(enabled=amp and dtype == torch.float16)
    best_loss, patience, logs = float("inf"), 0, []
    checkpoint = Path(out_dir) / "encoder_best.pt"
    accum = cfg["data"]["grad_accumulation"]
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(train_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            with _autocast(device, amp, dtype):
                out = model(images); loss = model.classification_loss(out, labels) / accum
            scaler.scale(loss).backward(); losses.append(float(loss) * accum)
            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step(); val_loss = _encoder_validation_loss(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        logs.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_loss": val_loss}, checkpoint)
        else:
            patience += 1
            if patience >= cfg["train"]["patience"]: break
    write_csv(logs, Path(out_dir) / "encoder_history.csv")
    state = torch.load(checkpoint, map_location=device); model.load_state_dict(state["model"])
    return model, state


def build_biomarker_clusters(train_rows, cfg, out_dir, num_classes, seed):
    values = np.asarray([[row[name] for name in BIOMARKER_COLUMNS] for row in train_rows], dtype=np.float64)
    standardized, median, iqr = robust_standardize(values, eps=cfg["prototypes"]["robust_epsilon"])
    labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    k = cfg["model"]["biomarker_prototypes_per_class"]
    assignments = np.empty(len(train_rows), dtype=np.int64); anchors = []
    for disease in range(num_classes):
        indices = np.flatnonzero(labels == disease)
        result = pam_kmedoids(standardized[indices], k, seed=seed + disease * 1000,
                              max_iter=cfg["prototypes"]["biomarker_max_iterations"],
                              repeats=cfg["prototypes"]["repeats"])
        assignments[indices] = result["assignments"]
        anchors.append(standardized[indices[result["medoid_indices"]]])
    clustered = [dict(row, bio_cluster=int(assignments[i])) for i, row in enumerate(train_rows)]
    anchors = np.concatenate(anchors).astype(np.float32)
    np.save(str(Path(out_dir) / "biomarker_anchors.npy"), anchors)
    atomic_json_dump({"columns": list(BIOMARKER_COLUMNS), "median": median.tolist(), "iqr": iqr.tolist(),
                      "epsilon": cfg["prototypes"]["robust_epsilon"], "assignments": assignments.tolist()},
                     Path(out_dir) / "biomarker_clusters.json")
    return clustered, anchors


@torch.no_grad()
def _global_features(model, rows, cfg, device):
    model.eval(); features, labels, clusters = [], [], []
    amp, dtype = _amp_settings(device)
    for batch in _loader(rows, cfg, False):
        with _autocast(device, amp, dtype): output = model(batch["image"].to(device))
        features.append(output["global"].float())
        labels.append(batch["label"].to(device)); clusters.append(batch["bio_cluster"].to(device))
    return torch.cat(features), torch.cat(labels), torch.cat(clusters)


def train_prototype_guided(model, train_rows, val_rows, cfg, device, out_dir, class_names):
    """Paper Eq. (1)-(2): epoch-fixed visual prototypes, recomputed between epochs."""
    parameters = [
        {"params": model.encoder.parameters(), "lr": cfg["train"]["backbone_lr"]},
        {"params": model.eas.parameters(), "lr": cfg["train"]["backbone_lr"]},
        {"params": list(model.global_head.parameters()) + list(model.attention_head.parameters()),
         "lr": cfg["train"]["head_lr"]},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["prototype_epochs"]
    scheduler = _scheduler(optimizer, epochs, cfg["train"]["warmup_epochs"], cfg["train"]["min_lr"])
    train_loader, val_loader = _loader(train_rows, cfg, True), _loader(val_rows, cfg)
    amp, dtype = _amp_settings(device); scaler = torch.cuda.amp.GradScaler(enabled=amp and dtype == torch.float16)
    k = cfg["model"]["biomarker_prototypes_per_class"]; accum = cfg["data"]["grad_accumulation"]
    best_loss, patience, logs = float("inf"), 0, []
    checkpoint = Path(out_dir) / "prototype_guided_best.pt"
    for epoch in range(epochs):
        features, labels, clusters = _global_features(model, train_rows, cfg, device)
        prototypes = recompute_biomarker_visual_prototypes(features, labels, clusters, len(class_names), k).to(device)
        model.train(); optimizer.zero_grad(set_to_none=True); losses, cls_losses, pm_losses = [], [], []
        for step, batch in enumerate(train_loader):
            images, y = batch["image"].to(device), batch["label"].to(device)
            cluster = batch["bio_cluster"].to(device)
            with _autocast(device, amp, dtype):
                output = model(images)
                cls = model.classification_loss(output, y)
                pm = prototype_matching_loss(output["global"], y, cluster, prototypes, k,
                                             cfg["train"]["temperature"])
                loss = (cls + cfg["train"]["lambda_bio"] * pm) / accum
            scaler.scale(loss).backward(); losses.append(float(loss) * accum)
            cls_losses.append(float(cls)); pm_losses.append(float(pm))
            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step(); val_loss = _encoder_validation_loss(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "classification_loss": float(np.mean(cls_losses)), "prototype_matching_loss": float(np.mean(pm_losses)),
               "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        logs.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_loss": val_loss}, checkpoint)
        else:
            patience += 1
            if patience >= cfg["train"]["patience"]: break
    write_csv(logs, Path(out_dir) / "prototype_guided_history.csv")
    state = torch.load(checkpoint, map_location=device); model.load_state_dict(state["model"])
    features, labels, clusters = _global_features(model, train_rows, cfg, device)
    prototypes = recompute_biomarker_visual_prototypes(features, labels, clusters, len(class_names), k)
    prototypes_np = prototypes.cpu().numpy().astype(np.float32)
    np.save(str(Path(out_dir) / "biomarker_prototypes.npy"), prototypes_np)
    return model, prototypes_np, state


@torch.no_grad()
def extract_pathology_prototypes(model, train_rows, cfg, device, out_dir, num_classes, seed):
    """Write all normalized layer3 tokens/weights to per-class memmaps, then cluster."""
    model.eval(); loader = _loader(train_rows, cfg, False)
    spatial = (cfg["data"]["input_size"] // 16) ** 2; dim = cfg["model"]["feature_dim"]
    counts_img = np.bincount([r["label"] for r in train_rows], minlength=num_classes)
    cache = ensure_dir(Path(out_dir) / "token_cache")
    token_maps, weight_maps, offsets = [], [], np.zeros(num_classes, dtype=np.int64)
    for c, image_count in enumerate(counts_img):
        n = int(image_count * spatial)
        token_maps.append(np.memmap(str(cache / "class{}_tokens.f16".format(c)), mode="w+", dtype=np.float16, shape=(n, dim)))
        weight_maps.append(np.memmap(str(cache / "class{}_weights.f32".format(c)), mode="w+", dtype=np.float32, shape=(n,)))
    amp, dtype = _amp_settings(device)
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].numpy()
        with _autocast(device, amp, dtype): out = model(images)
        tokens, weights = out["tokens"].float().cpu().numpy(), out["attention"].float().cpu().numpy()
        for i, c in enumerate(labels):
            start, end = offsets[c], offsets[c] + spatial
            token_maps[c][start:end] = tokens[i]; weight_maps[c][start:end] = weights[i]; offsets[c] = end
    centers, assignment_files = [], []
    for c in range(num_classes):
        token_maps[c].flush(); weight_maps[c].flush()
        cluster_fn = weighted_spherical_kmeans_torch if device.type == "cuda" else weighted_spherical_kmeans
        cluster_kwargs = dict(seed=seed + c * 1000, max_iter=cfg["prototypes"]["pathology_max_iterations"],
                              tol=cfg["prototypes"]["pathology_tolerance"], repeats=cfg["prototypes"]["repeats"],
                              chunk_size=max(cfg["prototypes"]["token_chunk_size"], 65536) if device.type == "cuda"
                              else cfg["prototypes"]["token_chunk_size"])
        if device.type == "cuda": cluster_kwargs["device"] = device
        result = cluster_fn(token_maps[c], weight_maps[c], cfg["model"]["pathology_prototypes_per_class"], **cluster_kwargs)
        centers.append(result["centers"])
        assignment_path = cache / "class{}_assignments.npy".format(c)
        np.save(str(assignment_path), result["assignments"]); assignment_files.append(str(assignment_path))
    prototypes = np.concatenate(centers, axis=0)
    np.save(str(Path(out_dir) / "pathology_prototypes.npy"), prototypes)
    atomic_json_dump({"class_counts": counts_img.tolist(), "tokens_per_image": spatial,
                      "assignment_files": assignment_files}, Path(out_dir) / "pathology_prototypes_meta.json")
    # Raw token/weight memmaps are reproducible temporary artifacts (roughly
    # 25-35 GiB per split). Keep compact assignments, but release these files
    # before starting the next independent experiment.
    token_maps.clear(); weight_maps.clear(); gc.collect()
    for pattern in ("class*_tokens.f16", "class*_weights.f32"):
        for temporary in cache.glob(pattern): temporary.unlink()
    return prototypes


@torch.no_grad()
def _trace_validation_loss(model, loader, device):
    """Validation used only for checkpoint selection during graph training."""
    model.eval(); losses = []
    amp, dtype = _amp_settings(device)
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        with _autocast(device, amp, dtype):
            logits = model(images)["logits"]
            losses.append(float(F.cross_entropy(logits, labels)))
    return float(np.mean(losses))


@torch.no_grad()
def _collect_coverage(model, loader, device):
    """Collect training-set evidence coverage for Tukey OOD calibration."""
    model.eval(); coverage = []
    amp, dtype = _amp_settings(device)
    for batch in loader:
        with _autocast(device, amp, dtype):
            output = model(batch["image"].to(device), apply_ood=False)
        coverage.extend(output["coverage"].float().cpu().tolist())
    return coverage


def train_graph(encoder_model, pathology_prototypes, biomarker_prototypes, biomarker_anchors,
                train_rows, val_rows, cfg, device, out_dir, class_names):
    topology = build_graph_topology(
        len(class_names), cfg["model"]["pathology_prototypes_per_class"], pathology_prototypes,
        k_bio=cfg["model"]["biomarker_prototypes_per_class"], biomarker_anchors=biomarker_anchors,
        full_mode=True)
    model = TRACEModel(len(class_names), pathology_prototypes, topology, pretrained=False,
                       biomarker_prototypes=biomarker_prototypes,
                       eas_hidden=cfg["model"]["eas_hidden_dim"], graph_hidden=cfg["model"]["graph_hidden_dim"],
                       graph_heads=cfg["model"]["graph_heads"], graph_layers=cfg["model"]["graph_layers"],
                       dropout=cfg["model"]["dropout"], match_epsilon=cfg["model"]["match_epsilon"]).to(device)
    model.encoder.load_state_dict(encoder_model.encoder.state_dict()); model.eas.load_state_dict(encoder_model.eas.state_dict())
    for p in model.encoder.parameters(): p.requires_grad_(False)
    for p in model.eas.parameters(): p.requires_grad_(False)
    params = list(model.hgt.parameters()) + [model.disease_embeddings] + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["graph_lr"], weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["graph_epochs"]; scheduler = _scheduler(optimizer, epochs, cfg["train"]["warmup_epochs"], cfg["train"]["min_lr"])
    train_loader, val_loader = _loader(train_rows, cfg, True), _loader(val_rows, cfg)
    amp, dtype = _amp_settings(device); scaler = torch.cuda.amp.GradScaler(enabled=amp and dtype == torch.float16)
    best_loss, patience, logs = float("inf"), 0, []; checkpoint = Path(out_dir) / "trace_best.pt"
    accum = cfg["data"]["grad_accumulation"]
    for epoch in range(epochs):
        model.train(); model.encoder.eval(); model.eas.eval(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step, batch in enumerate(train_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            with _autocast(device, amp, dtype): loss = F.cross_entropy(model(images)["logits"], labels) / accum
            scaler.scale(loss).backward(); losses.append(float(loss)*accum)
            if (step+1)%accum==0 or step+1==len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step(); val_loss = _trace_validation_loss(model, val_loader, device)
        row={"epoch":epoch+1,"train_loss":float(np.mean(losses)),
             "val_loss":val_loss,"lr":optimizer.param_groups[0]["lr"]}
        logs.append(row); print(json.dumps(row, ensure_ascii=False),flush=True)
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss,patience=val_loss,0
            torch.save({"model":model.state_dict(),"topology":topology.state_dict(),
                        "val_loss":val_loss},checkpoint)
        else:
            patience+=1
            if patience>=cfg["train"]["patience"]: break
    write_csv(logs,Path(out_dir)/"graph_history.csv"); state=torch.load(checkpoint,map_location=device); model.load_state_dict(state["model"])
    return model,state


def tukey_lower_fence(values, multiplier=1.5):
    q1, q3 = np.percentile(np.asarray(values, dtype=np.float64), [25, 75])
    return float(q1 - multiplier * (q3 - q1))


def run_experiment(cfg, experiment, device):
    if cfg["project"].get("mode") != "full":
        raise ValueError("Paper.pdf defines full dual-prototype TRACE only")
    protocec = cfg["model"].get("protocec_checkpoint")
    if not protocec or not Path(protocec).is_file():
        raise FileNotFoundError("Full TRACE training requires a provenance-bearing ProtoCEC checkpoint")
    seed = int(cfg["data"]["cv_seed"]) + int(str(experiment).replace("fold", "")) - 1
    seed_everything(seed)
    output=Path(cfg["project"]["output_dir"]); split_dir=output/"splits"/experiment; out=ensure_dir(output/"experiments"/experiment)
    train_rows=load_manifest(split_dir/"train.csv"); val_rows=load_manifest(split_dir/"val.csv"); test_rows=load_manifest(split_dir/"test.csv")
    if any(any(name not in row for name in BIOMARKER_COLUMNS) or not row.get("patient_id") for row in train_rows):
        raise ValueError("Training manifest lacks ProtoCEC biomarkers or patient IDs")
    class_names=sorted(cfg["data"]["included_classes"])
    atomic_json_dump(cfg,out/"config.json")
    encoder_checkpoint=out/"encoder_best.pt"
    if encoder_checkpoint.exists():
        encoder=EncoderPretrainingModel(len(class_names),pretrained=False,
                                        eas_hidden=cfg["model"]["eas_hidden_dim"]).to(device)
        encoder_state=torch.load(encoder_checkpoint,map_location=device);encoder.load_state_dict(encoder_state["model"])
        print("resumed_stage=encoder checkpoint={}".format(encoder_checkpoint),flush=True)
    else:
        encoder,encoder_state=train_encoder(train_rows,val_rows,cfg,device,out,class_names)
    clustered_rows, biomarker_anchors = build_biomarker_clusters(train_rows, cfg, out, len(class_names), seed)
    guided_checkpoint = out / "prototype_guided_best.pt"
    biomarker_file = out / "biomarker_prototypes.npy"
    if guided_checkpoint.exists() and biomarker_file.exists():
        guided_state = torch.load(guided_checkpoint, map_location=device)
        encoder.load_state_dict(guided_state["model"]); biomarker_prototypes = np.load(str(biomarker_file))
        print("resumed_stage=prototype_guided checkpoint={}".format(guided_checkpoint), flush=True)
    else:
        encoder, biomarker_prototypes, guided_state = train_prototype_guided(
            encoder, clustered_rows, val_rows, cfg, device, out, class_names)
    prototype_file=out/"pathology_prototypes.npy"
    if prototype_file.exists():
        pathology_prototypes=np.load(str(prototype_file));print("resumed_stage=pathology_prototypes file={}".format(prototype_file),flush=True)
    else:
        pathology_prototypes=extract_pathology_prototypes(encoder,train_rows,cfg,device,out,len(class_names),seed)
    model,graph_state=train_graph(encoder,pathology_prototypes,biomarker_prototypes,biomarker_anchors,
                                  train_rows,val_rows,cfg,device,out,class_names)
    coverage = _collect_coverage(model, _loader(train_rows, cfg), device)
    threshold=tukey_lower_fence(coverage,cfg["ood"]["multiplier"]); model.set_ood_threshold(threshold)
    result={"experiment":experiment,"seed":seed,"mode":"TRACE-Full",
            "ood_threshold":threshold,"train_count":len(train_rows),
            "val_count":len(val_rows),"test_count":len(test_rows),
            "val_loss":graph_state["val_loss"]}
    atomic_json_dump(result,out/"training_summary.json")
    torch.save({"model":model.state_dict(),"topology":graph_state["topology"],"class_names":class_names,
                "mode":"TRACE-Full","ood_threshold":threshold,"seed":seed,"config":cfg},out/"deployment.pt")
    return result
