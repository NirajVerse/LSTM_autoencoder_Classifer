"""LSTM autoencoder and hybrid attack classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAutoencoder(nn.Module):
    """Sequence autoencoder: (B, T, F) -> reconstruction + latent (B, L)."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)

        self.from_latent_h = nn.Linear(latent_size, hidden_size)
        self.from_latent_c = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, n_features)

    def encode_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent vector and per-timestep encoder outputs."""
        seq_out, (h_n, _) = self.encoder(x)
        z = self.to_latent(h_n[-1])
        return z, seq_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode_sequence(x)
        return z

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h0 = self.from_latent_h(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.from_latent_c(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        dec_in = torch.zeros(batch, seq_len, self.n_features, device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder(dec_in, (h0, c0))
        return self.output(dec_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, _ = self.encode_sequence(x)
        recon = self.decode(x, z)
        return recon, z


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block: x + Drop(W2(GELU(W1(LN(x)))))."""

    def __init__(self, dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class AttackClassifier(nn.Module):
    """
    Hybrid attack classifier.

    Inputs (per window):
      - x:       (B, T, F)  scaled raw + engineered KPI window
      - z:       (B, L)     frozen AE latent  (global summary)
      - seq_out: (B, T, H)  frozen AE encoder per-timestep outputs

    Pipeline:
      1. Trainable BiLSTM on top of frozen encoder outputs — gives the
         classifier its own bidirectional temporal model.
      2. Multi-head attention pooling over the BiLSTM outputs — captures
         multiple temporal aspects (onset, peak, variance regions, end).
      3. Global average of BiLSTM outputs — stable trend summary.
      4. Rich per-window stats over raw features: mean, std, min, max,
         last, slope.
      5. Concatenated features go through a projection + two residual MLP
         blocks with LayerNorm + GELU, then to the classification head.

    The AE is kept frozen; this module is the only trainable part.
    """

    def __init__(
        self,
        n_features: int,
        latent_size: int,
        encoder_hidden: int,
        n_classes: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
        bilstm_hidden: int = 32,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        self.n_features = n_features

        # 1. Trainable BiLSTM head over frozen encoder outputs.
        self.bilstm = nn.LSTM(
            input_size=encoder_hidden,
            hidden_size=bilstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        bilstm_out = bilstm_hidden * 2  # bidir

        # 2. Multi-head attention pooling.
        self.attn_heads = attn_heads
        self.attn_scores = nn.Linear(bilstm_out, attn_heads)
        self.attn_proj = nn.Linear(attn_heads * bilstm_out, bilstm_out)
        self.attn_norm = nn.LayerNorm(bilstm_out)

        # 4. Stats: 6 stats per feature.
        stats_per_feat = 6
        stats_dim = n_features * stats_per_feat

        # Combined feature dim: z + attn_pool + bilstm_global + stats
        in_dim = latent_size + bilstm_out + bilstm_out + stats_dim

        # 5. Projection + residual MLP head.
        proj_dim = hidden_sizes[0]
        self.proj = nn.Linear(in_dim, proj_dim)
        self.proj_norm = nn.LayerNorm(proj_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.res_block1 = ResidualMLPBlock(proj_dim, expansion=2, dropout=dropout)

        down_dim = hidden_sizes[1] if len(hidden_sizes) > 1 else proj_dim // 2
        self.down = nn.Linear(proj_dim, down_dim)
        self.down_norm = nn.LayerNorm(down_dim)
        self.down_drop = nn.Dropout(dropout)

        self.res_block2 = ResidualMLPBlock(down_dim, expansion=2, dropout=dropout)

        self.head = nn.Linear(down_dim, n_classes)

    @staticmethod
    def _window_stats(x: torch.Tensor) -> torch.Tensor:
        """Compute (mean, std, min, max, last, slope) across the time axis."""
        mu = x.mean(dim=1)
        sigma = x.std(dim=1, unbiased=False)
        xmin = x.min(dim=1).values
        xmax = x.max(dim=1).values
        xlast = x[:, -1, :]
        slope = (x[:, -1, :] - x[:, 0, :]) / float(x.size(1))
        return torch.cat([mu, sigma, xmin, xmax, xlast, slope], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        seq_out: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Bidirectional temporal model on top of frozen encoder outputs.
        bilstm_out, _ = self.bilstm(seq_out)  # (B, T, 2*H_b)

        # 2. Multi-head attention pooling.
        attn_logits = self.attn_scores(bilstm_out)               # (B, T, heads)
        attn_weights = F.softmax(attn_logits, dim=1)             # softmax over T
        head_contexts = torch.einsum(
            "bth,btd->bhd", attn_weights, bilstm_out
        )                                                        # (B, heads, D)
        flat = head_contexts.reshape(head_contexts.size(0), -1)  # (B, heads*D)
        attn_pool = self.attn_norm(self.attn_proj(flat))         # (B, D)

        # 3. Stable trend summary.
        bilstm_global = bilstm_out.mean(dim=1)                   # (B, D)

        # 4. Rich per-window statistics over the raw window.
        stats = self._window_stats(x)                            # (B, 6*F)

        # 5. Concatenate and run the residual MLP head.
        features = torch.cat(
            [z, attn_pool, bilstm_global, stats], dim=1
        )                                                        # (B, in_dim)

        h = self.proj(features)
        h = self.proj_norm(h)
        h = F.gelu(h)
        h = self.proj_drop(h)

        h = self.res_block1(h)

        h = self.down(h)
        h = self.down_norm(h)
        h = F.gelu(h)
        h = self.down_drop(h)

        h = self.res_block2(h)
        return self.head(h)


# Backward-compatible alias for any older code paths.
LatentClassifier = AttackClassifier
