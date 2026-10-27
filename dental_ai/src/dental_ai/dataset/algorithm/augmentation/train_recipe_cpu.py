from __future__ import annotations

from collections.abc import Callable

import torch

from .augmentation import AugmentationInput, AugmentationOutput, AugmentationParams, ScanSample
from .common.rng import chance
from .gingiva_trim_cpu import gingiva_trim_cpu
from .mirror_cpu import mirror_cpu
from .rotation_cpu import rotation_cpu
from .segment_crop_cpu import segment_crop_cpu
from .synth_colour_cpu import synth_colour_cpu
from .wedge_crop_cpu import wedge_crop_cpu


def _apply(fn: Callable[[AugmentationInput, AugmentationOutput], bool], sample: ScanSample,
           params: AugmentationParams, gen: torch.Generator | None) -> ScanSample:
    """fn을 적용한 sample, fn이 False를 내면 원래 sample."""
    res = AugmentationOutput()
    return res.sample if fn(AugmentationInput(sample, params, gen), res) else sample


def train_recipe_cpu(inp: AugmentationInput, out: AugmentationOutput) -> bool:
    """학습 augmentation 한 벌: mirror -> crop (악궁이 아직 원래 방향일 때) -> gingiva trim
    -> rotation -> colour.  단계마다 params의 확률로 돈다.

    colour_p 확률로 실제 색을 쓰거나 칠하고, 아니면 실제 색이 있어도 색 없이 넣는다 —
    network가 색에 기대지 않게.  crop과 trim이 사양해도 나머지는 돌므로 항상 True.
    """
    p, gen = inp.params, inp.generator
    sample = inp.sample
    if chance(gen, p.mirror_p):
        sample = _apply(mirror_cpu, sample, p, gen)
    if chance(gen, p.crop_p):
        crop = wedge_crop_cpu if chance(gen, p.wedge_p) else segment_crop_cpu
        sample = _apply(crop, sample, p, gen)
    if chance(gen, p.trim_p):
        sample = _apply(gingiva_trim_cpu, sample, p, gen)
    sample = _apply(rotation_cpu, sample, p, gen)
    if not chance(gen, p.colour_p):
        sample = sample.replace(colour=None)
    elif sample.colour is None:
        sample = _apply(synth_colour_cpu, sample, p, gen)
    out.sample = sample
    return True
