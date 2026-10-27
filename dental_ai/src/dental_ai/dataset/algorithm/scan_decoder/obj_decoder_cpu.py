from __future__ import annotations

import numpy as np
import torch

from .scan_decoder import ScanDecoderInput, ScanDecoderOutput


def obj_decoder_cpu(inp: ScanDecoderInput, out: ScanDecoderOutput) -> bool:
    """Wavefront OBJ: 'v x y z [r g b]'와 'f i j k' (segmentation3d obj decoder 포트).

    Teeth3DS 파일은 기계가 쓴 규칙적인 형식이라 블록마다 한 번 이어 붙여 numpy의 C
    tokenizer로 읽는다.  'f v/vt/vn'은 정점 index만 쓰고 'vn' / 'vt' 줄은 무시한다.
    False: 정점이 없음.
    """
    buf = inp.path.read_bytes()
    k = buf.find(b"\nf ")
    vb, fb = (buf, b"") if k < 0 else (buf[:k], buf[k + 1:])

    vlines = [ln for ln in vb.split(b"\n") if ln[:2] == b"v "]
    if not vlines:
        return False
    ncol = len(vlines[0].split())
    flat = np.fromstring(b" ".join(ln[2:] for ln in vlines), dtype=np.float32, sep=" ").reshape(len(vlines), ncol - 1)

    out.positions = torch.from_numpy(np.ascontiguousarray(flat[:, :3]))
    out.colours = torch.from_numpy(np.ascontiguousarray(flat[:, 3:6])) if flat.shape[1] >= 6 else None
    out.normals = out.face_colours = out.faces = out.fdi = out.labels = out.instances = None
    if fb:
        flines = [ln for ln in fb.split(b"\n") if ln[:2] == b"f "]
        body = b" ".join(ln[2:] for ln in flines)
        if b"/" in body:
            body = b" ".join(t.split(b"/", 1)[0] for t in body.split())
        faces = np.fromstring(body, dtype=np.int64, sep=" ").reshape(len(flines), -1)[:, :3] - 1
        out.faces = torch.from_numpy(np.ascontiguousarray(faces)) if len(faces) else None
    return True
