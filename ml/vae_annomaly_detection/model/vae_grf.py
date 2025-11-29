"""Implementation of the VAE-GRF model described in the paper."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder.decoder import Decoder
from model.encoder.encoder import Encoder


class VAEGRF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        latent_map_size: int,
        dropout: float = 0.1,
        corr_type: str = "corr_m32",
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        if latent_map_size <= 0:
            raise ValueError("latent_map_size must be positive")
        if latent_dim % (latent_map_size ** 2) != 0:
            raise ValueError("latent_dim must be divisible by latent_map_size**2 for VAE-GRF")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.latent_map_size = latent_map_size
        self.latent_channels = latent_dim // (latent_map_size ** 2)
        self.beta = beta

        self.encoder = Encoder(input_dim, hidden_dim, layers=3, dropout=dropout)
        enc_out = self.encoder.output_dim
        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, layers=3, dropout=dropout)

        self.corr_type = corr_type
        self.corr_fn = self._get_corr_fn(corr_type)
        self.logrange_prior = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.logsigma_prior = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.register_buffer("latent_distance", self._build_torus_distance(latent_map_size))

    def _get_corr_fn(self, corr_type: str):
        if corr_type == "corr_exp":
            return self._corr_exp
        if corr_type == "corr_gauss":
            return self._corr_gauss
        if corr_type == "corr_m32":
            return self._corr_matern32
        if corr_type == "corr_id":
            return self._corr_identity
        raise ValueError(f"Unsupported corr_type '{corr_type}'")

    def _build_torus_distance(self, size: int) -> torch.Tensor:
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij"), dim=-1).float()
        dx = torch.minimum(coords[..., 0], float(size) - coords[..., 0])
        dy = torch.minimum(coords[..., 1], float(size) - coords[..., 1])
        return torch.sqrt(dx ** 2 + dy ** 2)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h).view(-1, self.latent_channels, self.latent_map_size, self.latent_map_size)
        logvar = self.fc_logvar(h).view_as(mu)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z.view(z.size(0), -1))
        return recon, mu, logvar

    # -------------------- GRF utilities --------------------

    def _corr_exp(self, logsigma: torch.Tensor | None = None, logrange: torch.Tensor | None = None) -> torch.Tensor:
        range_ = torch.exp(self.logrange_prior if logrange is None else logrange)
        sigma2 = torch.exp(2 * (self.logsigma_prior if logsigma is None else logsigma))
        return sigma2 * torch.exp(-self.latent_distance / range_)

    def _corr_gauss(self, logsigma: torch.Tensor | None = None, logrange: torch.Tensor | None = None) -> torch.Tensor:
        range_ = torch.exp(self.logrange_prior if logrange is None else logrange)
        sigma2 = torch.exp(2 * (self.logsigma_prior if logsigma is None else logsigma))
        return sigma2 * torch.exp(-(self.latent_distance / range_) ** 2)

    def _corr_matern32(self, logsigma: torch.Tensor | None = None, logrange: torch.Tensor | None = None) -> torch.Tensor:
        range_ = torch.exp(self.logrange_prior if logrange is None else logrange)
        sigma2 = torch.exp(2 * (self.logsigma_prior if logsigma is None else logsigma))
        dist = self.latent_distance / range_
        return sigma2 * (1.0 + dist) * torch.exp(-dist)

    def _corr_identity(self, logsigma: torch.Tensor | None = None, logrange: torch.Tensor | None = None) -> torch.Tensor:
        del logsigma, logrange
        base = torch.zeros_like(self.latent_distance)
        base[0, 0] = 1.0
        return base

    def _get_base_covariance(self) -> torch.Tensor:
        cov = self.corr_fn().to(self.latent_distance.device)
        return cov.unsqueeze(0).unsqueeze(0)

    def _fft2(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(tensor, norm="ortho")

    def _ifft2(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(tensor, norm="ortho")

    def _matrix_vector_product(self, base: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        base_fft = self._fft2(base)
        value_fft = self._fft2(value)
        prod = torch.real(self._ifft2(base_fft * value_fft))
        scale = math.sqrt(self.latent_map_size ** 2)
        return prod * scale

    def _invert_base(self, base: torch.Tensor) -> torch.Tensor:
        base_fft = self._fft2(base)
        inv_fft = (1.0 / (self.latent_map_size ** 2)) * (1.0 / (base_fft + 1e-6))
        return torch.real(self._ifft2(inv_fft))

    def _logdet_base(self, base: torch.Tensor) -> torch.Tensor:
        base_fft = self._fft2(base)
        logdet = torch.sum(torch.log(torch.real(base_fft) + 1e-6), dim=(-2, -1))
        return logdet

    def kld(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = torch.exp(logvar)
        cov_base = self._get_base_covariance()
        inv_base = self._invert_base(cov_base)

        mu_flat = mu.view(mu.size(0), mu.size(1), -1)
        inv_mu = self._matrix_vector_product(inv_base, mu).view_as(mu_flat)

        trace_term = torch.sum(inv_base[:, :, 0, 0] * var, dim=(-1, -2))
        mahalanobis = torch.sum(inv_mu * mu_flat, dim=-1)
        logdet_prior = self._logdet_base(cov_base).squeeze()
        logdet_post = torch.sum(logvar, dim=(-1, -2))

        latent_area = float(self.latent_map_size ** 2)
        kld = 0.5 / latent_area * (
            -logdet_prior
            - mahalanobis
            - trace_term
            + logdet_post
            + latent_area
        )
        return torch.mean(kld, dim=1)

    def loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld_loss = torch.mean(self.kld(mu, logvar))
        total = recon_loss + self.beta * kld_loss
        return total, recon_loss, kld_loss
