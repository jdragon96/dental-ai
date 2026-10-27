"""색 없는 스캔에 치아 / 치은별로 칠하는 정점 색.  컬러맵은 in-house 실측값이다 (common/colour_map.py).

실측 스캔 하나를 골라 그 통계대로 칠한다: 치아 / 치은 평균색, 치아마다의 평균색 차이,
1~3 mm 길이로 완만하게 변하는 drift, channel끼리 함께 움직이는 작은 점별 noise.  실제
스캐너 색은 얼룩지지 않고 흘러간다 — 점별 noise는 drift의 절반 정도다.  치아와 치은이
겹치는 정도 (d')도 실측과 같아야 색이 답이 아니라 힌트로 남는다: 예전에 둘을 멀리
칠했더니 network가 label을 색에서 바로 읽어 버렸다 (val mAP 66.8 -> 38.7).
"""
from __future__ import annotations

import math

import torch

from .augmentation import AugmentationInput, AugmentationOutput
from .common.colour_map import (DRIFT_CHANNEL_CORR, GINGIVA_DRIFT_LENGTH_MM, GINGIVA_DRIFT_SD, GINGIVA_MEAN,
                                GINGIVA_NOISE_SD, NOISE_CHANNEL_CORR, SHIFT_CHANNEL_CORR, TOOTH_DRIFT_LENGTH_MM,
                                TOOTH_DRIFT_SD, TOOTH_MEAN, TOOTH_NOISE_SD, TOOTH_SHIFT_SD)
from .common.rng import chance, normal, randint

# 아말감, 금속, 연마 금속, 금, 지르코니아, 어두운 레진: "법랑질은 밝고 gingiva는 붉다"를 일부러
# 깬다.  in-house 스캔에는 label이 없어 수복물은 실측하지 못했다 — 문헌 값이다.
RESTORATION_SHADES = torch.tensor([
    (0.45, 0.46, 0.48), (0.62, 0.63, 0.66), (0.78, 0.79, 0.82), (0.83, 0.69, 0.35),
    (0.93, 0.92, 0.90), (0.70, 0.62, 0.52)])
RESTORATION_P = 0.35
FIELD_FEATURES = 16              # drift field 하나에 쓰는 random Fourier feature 수


def _correlated_normal(gen: torch.Generator | None, n: int, rho: float) -> torch.Tensor:
    """[n,3] 표준 정규, channel 사이 상관 rho."""
    return math.sqrt(rho) * normal(gen, (n, 1), 1.0) + math.sqrt(1 - rho) * normal(gen, (n, 3), 1.0)


def _smooth_field(gen: torch.Generator | None, q: torch.Tensor, length_mm: float, rho: float) -> torch.Tensor:
    """[n,3] 분산 1인 Gaussian 공간 field: 공분산 exp(-r² / 2ℓ²) (random Fourier features), channel 상관 rho."""
    def field() -> torch.Tensor:
        w = normal(gen, (3, FIELD_FEATURES), 1.0 / length_mm)
        phase = torch.rand(FIELD_FEATURES, generator=gen) * (2 * math.pi)
        return math.sqrt(2.0 / FIELD_FEATURES) * torch.cos(q @ w + phase).sum(1)

    shared = field()
    return math.sqrt(rho) * shared[:, None] + math.sqrt(1 - rho) * torch.stack([field() for _ in range(3)], 1)


def synth_colour_cpu(inp: AugmentationInput, out: AugmentationOutput) -> bool:
    """sample.colour를 칠한 색 [V,3] 0..1로 바꾼다.  치아 / 치은은 classes > 0으로 가른다."""
    s, gen = inp.sample, inp.generator
    positions, classes, instances = s.positions, s.classes, s.instances
    n = positions.shape[0]
    t = classes > 0
    k = randint(gen, 0, TOOTH_MEAN.shape[0])                 # 실측 스캔 하나

    c = torch.where(t[:, None], TOOTH_MEAN[k], GINGIVA_MEAN[k])

    # 한 입 안의 치아도 평균색이 조금씩 다르다
    ids = torch.unique(instances)
    ids = ids[ids > 0]
    tooth = instances > 0
    slot = torch.searchsorted(ids, instances[tooth])
    c[tooth] += (_correlated_normal(gen, ids.numel(), SHIFT_CHANNEL_CORR) * TOOTH_SHIFT_SD[k])[slot]

    # 완만한 drift와 점별 noise, 실측한 크기와 상관 길이로
    q = positions - positions.mean(0)
    c[t] += _smooth_field(gen, q[t], TOOTH_DRIFT_LENGTH_MM, DRIFT_CHANNEL_CORR) * TOOTH_DRIFT_SD[k]
    c[~t] += _smooth_field(gen, q[~t], GINGIVA_DRIFT_LENGTH_MM, DRIFT_CHANNEL_CORR) * GINGIVA_DRIFT_SD[k]
    c += _correlated_normal(gen, n, NOISE_CHANNEL_CORR) * torch.where(t[:, None], TOOTH_NOISE_SD[k],
                                                                      GINGIVA_NOISE_SD[k])

    # 수복물: 법랑질이 아니라 금속·금·지르코니아인 크라운
    if ids.numel() and chance(gen, RESTORATION_P):
        count = min(ids.numel(), randint(gen, 1, 3))
        for j in ids[torch.randperm(ids.numel(), generator=gen)[:count]].tolist():
            m = instances == j
            shade = RESTORATION_SHADES[randint(gen, 0, len(RESTORATION_SHADES))] + normal(gen, (3,), 0.03)
            c[m] = shade + _correlated_normal(gen, int(m.sum()), NOISE_CHANNEL_CORR) * TOOTH_NOISE_SD[k]

    out.sample = s.replace(colour=c.clamp(0.0, 1.0))
    return True
