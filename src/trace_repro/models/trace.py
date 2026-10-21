import torch
from torch import nn
import torch.nn.functional as F

from ..prototypes import match_and_activate
from .encoder import SharedEncoder
from .eas import EvidenceAttentionScorer
from .hgt import HeterogeneousGraphTransformer


class TRACEModel(nn.Module):
    def __init__(self, num_classes, pathology_prototypes, topology, pretrained=True,
                 biomarker_prototypes=None, eas_hidden=256, graph_hidden=256,
                 graph_heads=4, graph_layers=2, dropout=0.1, match_epsilon=1e-6,
                 encoder_checkpoint=None):
        super().__init__()
        self.num_classes = num_classes
        self.full_mode = biomarker_prototypes is not None
        if self.full_mode != topology.full_mode: raise ValueError("Prototype mode/topology mismatch")
        self.encoder = SharedEncoder(pretrained=pretrained, checkpoint=encoder_checkpoint)
        self.eas = EvidenceAttentionScorer(1024, eas_hidden)
        self.disease_embeddings = nn.Parameter(torch.empty(num_classes, 1024))
        nn.init.normal_(self.disease_embeddings, std=0.02)
        self.register_buffer("pathology_prototypes", F.normalize(torch.as_tensor(pathology_prototypes).float(), dim=-1))
        if self.full_mode:
            self.register_buffer("biomarker_prototypes", F.normalize(torch.as_tensor(biomarker_prototypes).float(), dim=-1))
        else:
            self.register_buffer("biomarker_prototypes", torch.empty(0, 1024))
        self.register_buffer("node_types", topology.node_types)
        self.edge_names = tuple(topology.relation_names)
        for idx, name in enumerate(self.edge_names): self.register_buffer("edge_{}".format(idx), topology.edge_index[name])
        self.hgt = HeterogeneousGraphTransformer(1024, graph_hidden, graph_heads, graph_layers,
                                                  self.edge_names, num_node_types=3, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(graph_hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.match_epsilon = match_epsilon
        self.register_buffer("ood_threshold", torch.tensor(float("-inf")))

    @property
    def edge_index(self): return {name: getattr(self, "edge_{}".format(i)) for i, name in enumerate(self.edge_names)}

    def set_ood_threshold(self, threshold): self.ood_threshold.fill_(float(threshold))

    def encode(self, images):
        feature_map, global_desc, tokens = self.encoder(images)
        attention, attention_desc = self.eas(tokens)
        return {"feature_map": feature_map, "global": global_desc, "tokens": tokens,
                "attention": attention, "attention_descriptor": attention_desc}

    def forward(self, images, apply_ood=False):
        encoded = self.encode(images); batch = images.shape[0]
        path_activation, path_confidence, path_indices = match_and_activate(
            encoded["tokens"], encoded["attention"], self.pathology_prototypes, self.match_epsilon)
        path_nodes = path_activation.unsqueeze(-1) * self.pathology_prototypes.unsqueeze(0)
        coverage_path = path_activation.sum(dim=1)
        disease_nodes = self.disease_embeddings.unsqueeze(0).expand(batch, -1, -1)
        nodes = [disease_nodes]
        output = {"path_activation": path_activation, "path_indices": path_indices,
                  "path_confidence": path_confidence, "coverage_path": coverage_path,
                  "attention": encoded["attention"]}
        if self.full_mode:
            bio_activation, bio_confidence, bio_indices = match_and_activate(
                encoded["global"].unsqueeze(1), torch.ones(batch, 1, device=images.device),
                self.biomarker_prototypes, self.match_epsilon)
            nodes.append(bio_activation.unsqueeze(-1) * self.biomarker_prototypes.unsqueeze(0))
            coverage_bio = bio_activation.sum(1)
            coverage = torch.sqrt(torch.clamp(coverage_bio * coverage_path, min=0.0))
            output.update(bio_activation=bio_activation, bio_indices=bio_indices,
                          bio_confidence=bio_confidence, coverage_bio=coverage_bio)
        else:
            coverage = coverage_path
        nodes.append(path_nodes)
        graph_nodes = torch.cat(nodes, dim=1)
        is_ood = coverage < self.ood_threshold
        accepted = ~is_ood if apply_ood else torch.ones_like(is_ood, dtype=torch.bool)
        # Deployment gating happens before HGT: rejected samples finish after
        # encoder/EAS/prototype matching.
        logits = torch.full((batch, self.num_classes), float("nan"), device=images.device,
                            dtype=graph_nodes.dtype)
        if torch.any(accepted):
            graph_output = self.hgt(graph_nodes[accepted], self.node_types, self.edge_index)
            accepted_logits = self.classifier(graph_output[:, :self.num_classes]).squeeze(-1)
            logits[accepted] = accepted_logits.to(logits.dtype)
        probabilities = torch.zeros_like(logits)
        if torch.any(accepted): probabilities[accepted] = torch.softmax(logits[accepted], dim=-1)
        prediction = torch.full((batch,), -1, device=images.device, dtype=torch.long)
        if torch.any(accepted): prediction[accepted] = logits[accepted].argmax(1)
        output.update(logits=logits, probabilities=probabilities, coverage=coverage,
                      is_ood=is_ood, prediction=prediction)
        return output


class EncoderPretrainingModel(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, encoder_checkpoint=None, eas_hidden=256):
        super().__init__()
        self.encoder = SharedEncoder(pretrained=pretrained, checkpoint=encoder_checkpoint)
        self.eas = EvidenceAttentionScorer(1024, eas_hidden)
        self.global_head = nn.Linear(1024, num_classes)
        self.attention_head = nn.Linear(1024, num_classes)

    def forward(self, images):
        _, global_desc, tokens = self.encoder(images)
        attention, attention_desc = self.eas(tokens)
        return {"global": global_desc, "tokens": tokens, "attention": attention,
                "global_logits": self.global_head(global_desc),
                "attention_logits": self.attention_head(attention_desc)}

    @staticmethod
    def classification_loss(output, labels):
        return 0.5 * (F.cross_entropy(output["global_logits"], labels) +
                      F.cross_entropy(output["attention_logits"], labels))
