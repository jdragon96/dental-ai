"""스캔 decoder의 Input / Output."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import torch


class VertexLabel(IntEnum):
    """data/ios raw PLY의 label: 정점이 무엇인가.  DentalLabeler의 클래스 (LabelTable.h)와 같은 값이다.
    새 범주는 뒤에 번호를 이어 붙인다."""

    UNLABELED = -1
    GINGIVA = 0
    TOOTH = 1
    SCAN_BODY = 2


@dataclass
class ScanDecoderInput:
    path: Path


@dataclass
class ScanDecoderOutput:
    positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))    # [V,3] float32 mm
    faces: torch.Tensor | None = None                                             # [F,3] int64
    colours: torch.Tensor | None = None                                           # [V,3] float32 0..1
    normals: torch.Tensor | None = None                                           # [V,3] float32, 파일에 적힌 그대로
    face_colours: torch.Tensor | None = None      # [F,3] uint8 — 3D-IOSSeg는 label을 face 색으로 저장한다
    # data/ios raw PLY의 정점 label (scan_encoder가 쓴다).  label 없는 스캔은 셋 다 None
    fdi: torch.Tensor | None = None               # [V] int64: 치아 번호 11..48, 위치가 없으면 0
    labels: torch.Tensor | None = None            # [V] int64: VertexLabel (-1 미분류, 0 치은, 1 치아, 2 스캔바디 ...)
    instances: torch.Tensor | None = None         # [V] int64: 치아 instance 번호 1..K, 치아가 아니면 0
