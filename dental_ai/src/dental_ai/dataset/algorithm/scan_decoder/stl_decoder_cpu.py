from __future__ import annotations

import numpy as np
import torch

from .scan_decoder import ScanDecoderInput, ScanDecoderOutput


def stl_decoder_cpu(inp: ScanDecoderInput, out: ScanDecoderOutput) -> bool:
    """STL (binary / ascii) -> 정점을 용접한 mesh.  face 순서는 파일 그대로다 (face label이 맞물린다).

    STL은 삼각형마다 정점 셋을 따로 적는다.  같은 비트의 정점을 하나로 합쳐야 면적 가중
    normal이 매끈해지고 정점 label을 붙일 수 있다.  False: 삼각형이 없음.
    """
    buf = inp.path.read_bytes()
    n = int(np.frombuffer(buf, "<u4", 1, 80)[0]) if len(buf) >= 84 else -1
    if n >= 0 and 84 + 50 * n == len(buf):                      # binary: 크기가 정확히 맞는다
        rec = np.frombuffer(buf, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
        corners = np.ascontiguousarray(rec["v"]).reshape(-1, 3)
    else:                                                       # ascii: 'vertex x y z' 줄만 읽는다
        lines = [ln.strip()[7:] for ln in buf.split(b"\n") if ln.lstrip().startswith(b"vertex ")]
        corners = np.fromstring(b" ".join(lines), dtype=np.float32, sep=" ").reshape(-1, 3)
    if len(corners) == 0 or len(corners) % 3:
        return False
    positions, faces = weld(torch.from_numpy(corners), torch.arange(len(corners)).reshape(-1, 3))
    out.positions, out.faces = positions, faces
    out.colours = out.normals = out.face_colours = out.fdi = out.labels = out.instances = None
    return True


def weld(positions: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """비트가 같은 정점을 합친다 (segmentation3d merge_duplicates 포트).  정점은 처음 나온 순서다.

    float 비트를 값 대신 쓴다: x|y를 int64 하나, z를 다른 하나로 묶고 안정 정렬 두 번으로
    줄 세운다 — float 행에 torch.unique(dim=0)을 쓰는 것보다 ~10배 빠르다.
    """
    bits = positions.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    xy, z = (bits[:, 0] << 32) | bits[:, 1], bits[:, 2]
    order = torch.argsort(z, stable=True)
    order = order[torch.argsort(xy[order], stable=True)]
    new = torch.ones(order.numel(), dtype=torch.bool)
    new[1:] = (xy[order][1:] != xy[order][:-1]) | (z[order][1:] != z[order][:-1])
    group = torch.cumsum(new, 0) - 1
    first = torch.full((int(group[-1]) + 1,), order.numel(), dtype=torch.int64)
    first.scatter_reduce_(0, group, order, "amin")              # 무리마다 가장 먼저 나온 정점
    rank = torch.empty_like(first)
    rank[torch.argsort(first)] = torch.arange(first.numel())
    inverse = torch.empty_like(order)
    inverse[order] = rank[group]
    return positions[torch.sort(first).values], inverse[faces]
