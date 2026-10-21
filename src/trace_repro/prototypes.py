import math

import numpy as np
import torch
import torch.nn.functional as F


def robust_standardize(values, median=None, iqr=None, eps=1e-6):
    values = np.asarray(values, dtype=np.float64)
    if median is None:
        median = np.nanmedian(values, axis=0)
    if iqr is None:
        q75, q25 = np.nanpercentile(values, [75, 25], axis=0)
        iqr = q75 - q25
    return (values - median) / (iqr + eps), median, iqr


def _pairwise_euclidean(a, b):
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))


def pam_kmedoids(values, k, seed=0, max_iter=300, repeats=5):
    """Deterministic multi-start PAM-style k-medoids for per-class biomarkers."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) < k:
        raise ValueError("K-medoids needs at least k samples")
    dist = _pairwise_euclidean(x, x)
    best = None
    for repeat in range(repeats):
        rng = np.random.RandomState(seed + repeat)
        medoids = [rng.randint(len(x))]
        closest = dist[:, medoids[0]].copy()
        for _ in range(1, k):
            probs = closest / max(closest.sum(), 1e-12)
            nxt = rng.choice(len(x), p=probs)
            while nxt in medoids:
                nxt = rng.choice(len(x))
            medoids.append(int(nxt)); closest = np.minimum(closest, dist[:, nxt])
        medoids = np.asarray(medoids)
        for _ in range(max_iter):
            assign = dist[:, medoids].argmin(axis=1)
            new = medoids.copy()
            for j in range(k):
                members = np.flatnonzero(assign == j)
                if len(members):
                    new[j] = members[np.argmin(dist[np.ix_(members, members)].sum(axis=1))]
            if np.array_equal(new, medoids): break
            medoids = new
        assign = dist[:, medoids].argmin(axis=1)
        cost = dist[np.arange(len(x)), medoids[assign]].sum()
        if best is None or cost < best[0]: best = (cost, medoids.copy(), assign.copy())
    return {"medoid_indices": best[1], "assignments": best[2], "cost": float(best[0])}


def recompute_biomarker_visual_prototypes(features, labels, clusters, num_classes, k):
    output = []
    for disease in range(num_classes):
        for cluster in range(k):
            mask = (labels == disease) & (clusters == cluster)
            if not torch.any(mask):
                raise ValueError("Empty biomarker cluster disease={} cluster={}".format(disease, cluster))
            output.append(F.normalize(features[mask].mean(0), dim=0))
    return torch.stack(output).detach()


def prototype_matching_loss(global_features, labels, clusters, prototypes, k, temperature=0.07):
    shaped = prototypes.view(-1, k, prototypes.shape[-1])
    class_proto = shaped[labels]
    logits = torch.einsum("bd,bkd->bk", F.normalize(global_features, dim=-1), class_proto) / temperature
    return F.cross_entropy(logits, clusters)


def weighted_spherical_kmeans(tokens, weights, k, seed=0, max_iter=100, tol=1e-4, repeats=5,
                              chunk_size=8192):
    """EAS-weighted spherical K-means, supporting ndarray/memmap inputs."""
    n, dim = tokens.shape
    if n < k: raise ValueError("Spherical K-means needs at least k tokens")
    best = None
    for repeat in range(repeats):
        rng = np.random.RandomState(seed + repeat)
        first = rng.randint(n); centers = [np.asarray(tokens[first], dtype=np.float64)]
        centers[0] /= max(np.linalg.norm(centers[0]), 1e-12)
        min_dist = np.full(n, np.inf)
        for _ in range(1, k):
            last = centers[-1]
            sims = np.empty(n)
            for start in range(0, n, chunk_size):
                chunk = np.asarray(tokens[start:start + chunk_size], dtype=np.float64)
                sims[start:start + len(chunk)] = chunk.dot(last)
            min_dist = np.minimum(min_dist, np.maximum(0.0, 1.0 - sims))
            probs = min_dist / max(min_dist.sum(), 1e-12)
            center = np.asarray(tokens[rng.choice(n, p=probs)], dtype=np.float64)
            centers.append(center / max(np.linalg.norm(center), 1e-12))
        centers = np.stack(centers)
        assignments = np.zeros(n, dtype=np.int32)
        previous = None
        for _ in range(max_iter):
            sums = np.zeros((k, dim), dtype=np.float64); objective = 0.0
            for start in range(0, n, chunk_size):
                chunk = np.asarray(tokens[start:start + chunk_size], dtype=np.float64)
                sims = chunk.dot(centers.T); assign = sims.argmax(axis=1)
                assignments[start:start + len(chunk)] = assign
                w = np.asarray(weights[start:start + len(chunk)], dtype=np.float64)
                objective += float(np.sum(w * sims[np.arange(len(chunk)), assign]))
                for j in range(k):
                    mask = assign == j
                    if np.any(mask): sums[j] += (chunk[mask] * w[mask, None]).sum(axis=0)
            norms = np.linalg.norm(sums, axis=1, keepdims=True)
            empty = norms[:, 0] < 1e-12
            if np.any(empty):
                for j in np.flatnonzero(empty): sums[j] = tokens[rng.randint(n)]
                norms = np.linalg.norm(sums, axis=1, keepdims=True)
            new_centers = sums / np.maximum(norms, 1e-12)
            shift = np.max(1.0 - np.sum(centers * new_centers, axis=1))
            centers = new_centers
            if shift < tol: break
            previous = objective
        result = (objective, centers.astype(np.float32), assignments.copy())
        if best is None or result[0] > best[0]: best = result
    return {"objective": float(best[0]), "centers": best[1], "assignments": best[2]}


def weighted_spherical_kmeans_torch(tokens, weights, k, device, seed=0, max_iter=100,
                                    tol=1e-4, repeats=5, chunk_size=65536):
    """Same full-token objective as above, accelerated on CUDA when available.

    The class-specific float16 memmap is copied to the GPU once. Assignments use
    cosine similarity; center updates accumulate weighted sums in FP32 chunks.
    """
    device = torch.device(device); n, dim = tokens.shape
    if n < k: raise ValueError("Spherical K-means needs at least k tokens")
    x = torch.from_numpy(np.asarray(tokens)).to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    w = torch.from_numpy(np.asarray(weights)).to(device=device, dtype=torch.float32)
    best = None
    for repeat in range(repeats):
        generator = torch.Generator(device=device); generator.manual_seed(seed + repeat)
        first = int(torch.randint(n, (1,), generator=generator, device=device))
        centers = [F.normalize(x[first].float(), dim=0)]
        min_dist = torch.full((n,), float("inf"), device=device)
        for _ in range(1, k):
            sims = torch.mv(x, centers[-1].to(x.dtype)).float()
            min_dist = torch.minimum(min_dist, torch.clamp(1.0 - sims, min=0.0))
            probs = min_dist / torch.clamp(min_dist.sum(), min=1e-12)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
            centers.append(F.normalize(x[nxt].float(), dim=0))
        centers = torch.stack(centers)
        assignments = torch.empty(n, dtype=torch.int64, device=device)
        objective = -float("inf")
        for _ in range(max_iter):
            sums = torch.zeros(k, dim, dtype=torch.float32, device=device)
            objective_tensor = torch.zeros((), dtype=torch.float64, device=device)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n); chunk = x[start:end]
                sims = torch.mm(chunk, centers.t().to(chunk.dtype)).float()
                best_sim, assign = sims.max(dim=1); assignments[start:end] = assign
                wc = w[start:end]
                objective_tensor += (wc.double() * best_sim.double()).sum()
                membership = F.one_hot(assign, num_classes=k).float() * wc[:, None]
                sums += torch.mm(membership.t(), chunk.float())
            norms = sums.norm(dim=1, keepdim=True); empty = norms[:, 0] < 1e-12
            if torch.any(empty):
                random_idx = torch.randint(n, (int(empty.sum()),), generator=generator, device=device)
                sums[empty] = x[random_idx].float(); norms = sums.norm(dim=1, keepdim=True)
            new_centers = sums / torch.clamp(norms, min=1e-12)
            shift = float(torch.max(1.0 - torch.sum(centers * new_centers, dim=1)))
            centers = new_centers; objective = float(objective_tensor)
            if shift < tol: break
        result = (objective, centers.cpu().numpy().astype(np.float32), assignments.cpu().numpy().astype(np.int32))
        if best is None or result[0] > best[0]: best = result
    del x, w
    if device.type == "cuda": torch.cuda.empty_cache()
    return {"objective": float(best[0]), "centers": best[1], "assignments": best[2]}


def match_and_activate(queries, query_weights, prototypes, epsilon=1e-6):
    queries = F.normalize(queries, dim=-1); prototypes = F.normalize(prototypes, dim=-1)
    similarity = torch.einsum("bnd,pd->bnp", queries, prototypes)
    best_sim, best_idx = similarity.max(dim=-1)
    confidence = epsilon + (1.0 - epsilon) * (1.0 + best_sim) / 2.0
    contribution = query_weights * confidence
    activation = torch.zeros(queries.shape[0], prototypes.shape[0], device=queries.device,
                             dtype=queries.dtype)
    activation.scatter_add_(1, best_idx, contribution)
    return activation, confidence, best_idx
