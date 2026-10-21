from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphTopology:
    num_classes: int
    k_path: int
    k_bio: int
    full_mode: bool
    node_types: torch.Tensor
    edge_index: dict
    relation_names: tuple

    @property
    def num_nodes(self): return int(self.node_types.numel())

    def to(self, device):
        return GraphTopology(self.num_classes, self.k_path, self.k_bio, self.full_mode,
                             self.node_types.to(device), {k: v.to(device) for k, v in self.edge_index.items()},
                             self.relation_names)

    def state_dict(self):
        return {"num_classes": self.num_classes, "k_path": self.k_path, "k_bio": self.k_bio,
                "full_mode": self.full_mode, "node_types": self.node_types,
                "edge_index": self.edge_index, "relation_names": self.relation_names}


def _mutual_cross_class_nearest(vectors, owners, metric="cosine"):
    vectors = np.asarray(vectors, dtype=np.float64); owners = np.asarray(owners)
    if metric == "cosine":
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        score = vectors.dot(vectors.T)
        np.fill_diagonal(score, -np.inf)
        score[owners[:, None] == owners[None, :]] = -np.inf
        nearest = score.argmax(axis=1)
    elif metric == "euclidean":
        score = ((vectors[:, None] - vectors[None, :]) ** 2).sum(-1)
        np.fill_diagonal(score, np.inf)
        score[owners[:, None] == owners[None, :]] = np.inf
        nearest = score.argmin(axis=1)
    else:
        raise ValueError("Unknown MNN metric: {}".format(metric))
    pairs = []
    for i, j in enumerate(nearest):
        if nearest[j] == i and i < j: pairs.append((i, int(j)))
    return pairs


def _edges(pairs):
    return torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty(2, 0, dtype=torch.long)


def build_graph_topology(num_classes, k_path, pathology_prototypes, k_bio=0,
                         biomarker_anchors=None, full_mode=False):
    bio_count = num_classes * k_bio if full_mode else 0
    path_count = num_classes * k_path
    bio_start = num_classes; path_start = num_classes + bio_count
    node_types = torch.tensor([0] * num_classes + ([1] * bio_count if full_mode else []) + [2] * path_count)
    relations = {}
    disease_anchor = []
    for d in range(num_classes):
        for k in range(k_path):
            node = path_start+d*k_path+k
            disease_anchor.extend(((d, node), (node, d)))
    if full_mode:
        cross_evidence = []
        for d in range(num_classes):
            bs = [bio_start+d*k_bio+k for k in range(k_bio)]
            ps = [path_start+d*k_path+k for k in range(k_path)]
            for b in bs: disease_anchor.extend(((d, b), (b, d)))
            for b in bs:
                for p in ps: cross_evidence.extend(((b, p), (p, b)))
        relations["disease_anchor"] = _edges(disease_anchor)
        relations["cross_evidence"] = _edges(cross_evidence)
        owners = np.repeat(np.arange(num_classes), k_bio)
        pairs = _mutual_cross_class_nearest(np.asarray(biomarker_anchors), owners, metric="euclidean")
        directed = [(bio_start+a,bio_start+b) for a,b in pairs] + [(bio_start+b,bio_start+a) for a,b in pairs]
        relations["shared_bio"] = _edges(directed)
    else:
        relations["disease_anchor"] = _edges(disease_anchor)
    owners = np.repeat(np.arange(num_classes), k_path)
    proto = np.asarray(pathology_prototypes, dtype=np.float64)
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    pairs = _mutual_cross_class_nearest(proto, owners, metric="cosine")
    directed = [(path_start+a,path_start+b) for a,b in pairs] + [(path_start+b,path_start+a) for a,b in pairs]
    relations["shared_path"] = _edges(directed)
    return GraphTopology(num_classes, k_path, k_bio, full_mode, node_types, relations, tuple(relations))

