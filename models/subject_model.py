"""Models that aggregate multiple EEG segment embeddings per subject."""

from contextlib import nullcontext

import torch
import torch.nn as nn


class SubjectAggregationModel(nn.Module):
    """Encode segments, optionally transform them, then mean-pool per subject."""

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int = 512,
        n_classes: int = 2,
        aggregation: str = "mean",
        transform_hidden_dim: int = 256,
        transform_dropout: float = 0.2,
        freeze_encoder: bool = True,
        encoder_chunk_size: int = 64,
    ):
        super().__init__()
        if aggregation not in {"mean", "transform_mean"}:
            raise ValueError("aggregation must be mean or transform_mean")
        if encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be positive")

        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.freeze_encoder = freeze_encoder
        self.encoder_chunk_size = encoder_chunk_size

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        if aggregation == "mean":
            self.transform = nn.Identity()
        else:
            self.transform = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, transform_hidden_dim),
                nn.GELU(),
                nn.Dropout(transform_dropout),
                nn.Linear(transform_hidden_dim, embedding_dim),
            )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode_valid_segments(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        context = torch.no_grad if self.freeze_encoder else nullcontext
        for start in range(0, eeg.shape[0], self.encoder_chunk_size):
            stop = start + self.encoder_chunk_size
            with context():
                embedding = self.encoder(eeg[start:stop], pos=pos[start:stop])
            if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
                raise ValueError(
                    "encoder must return a (n_segments, embedding_dim) tensor"
                )
            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"encoder returned {embedding.shape[1]} features; "
                    f"expected {self.embedding_dim}"
                )
            outputs.append(embedding)
        return torch.cat(outputs, dim=0)

    def encode_subjects(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one pooled embedding per subject."""
        if eeg.ndim != 4 or pos.ndim != 4 or segment_mask.ndim != 2:
            raise ValueError(
                "expected eeg (B,S,C,T), pos (B,S,C,3), mask (B,S)"
            )
        if eeg.shape[:2] != segment_mask.shape or pos.shape[:2] != segment_mask.shape:
            raise ValueError("segment dimensions and mask do not match")
        if not torch.all(segment_mask.any(dim=1)):
            raise ValueError("every subject must contain at least one segment")

        batch_size, max_segments = segment_mask.shape
        valid_eeg = eeg[segment_mask]
        valid_pos = pos[segment_mask]
        embeddings = self._encode_valid_segments(valid_eeg, valid_pos)
        transformed = self.transform(embeddings)

        subject_index = (
            torch.arange(batch_size, device=eeg.device)
            .unsqueeze(1)
            .expand(batch_size, max_segments)[segment_mask]
        )
        pooled = transformed.new_zeros(batch_size, transformed.shape[1])
        pooled.index_add_(0, subject_index, transformed)
        counts = torch.bincount(
            subject_index, minlength=batch_size
        ).to(transformed.dtype).unsqueeze(1)
        return pooled / counts

    def forward(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.encode_subjects(eeg, pos, segment_mask))
