from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .scan_decoder import ScanDecoderInput, ScanDecoderOutput

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_RGB_NAMES = (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue"))


@dataclass
class _Element:
    name: str
    count: int
    # header 순서대로: scalar는 (이름, type), list는 (이름, count type, item type)
    props: list[tuple[str, ...]] = field(default_factory=list)


def ply_decoder_cpu(inp: ScanDecoderInput, out: ScanDecoderOutput) -> bool:
    """Stanford PLY: binary (양쪽 endian) 또는 ascii, mesh 또는 point cloud.

    segmentation3d PLY decoder 포트에 face 속성을 더했다 — 3D-IOSSeg는 label을 face 색
    (list 뒤의 red / green / blue / alpha)으로 저장한다.  Medit / L-SCAN binary mesh
    (색, normal), CloudCompare ascii point cloud, data/ios raw PLY (정점 속성 fdi / label / instance)도
    읽는다.  False: 정점이 없거나 PLY가 아님.
    """
    buf = inp.path.read_bytes()
    end = buf.find(b"end_header")
    if not buf.startswith(b"ply") or end < 0:
        return False
    body_start = buf.index(b"\n", end) + 1
    fmt, elements = _parse_header(buf[:end].decode("ascii", "replace"))
    tables = (_read_ascii(buf[body_start:], elements) if fmt == "ascii"
              else _read_binary(buf, body_start, elements, "<" if fmt == "binary_little_endian" else ">"))

    vert = tables.get("vertex", {})
    if len(vert.get("x", ())) == 0:
        return False
    out.positions = torch.from_numpy(np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float32))
    out.normals = (torch.from_numpy(np.stack([vert["nx"], vert["ny"], vert["nz"]], 1).astype(np.float32))
                   if all(k in vert for k in ("nx", "ny", "nz")) else None)
    rgb = _rgb(vert)
    out.colours = None if rgb is None else torch.from_numpy(
        rgb.astype(np.float32) / 255.0 if rgb.dtype.kind in "iu" else rgb.astype(np.float32))
    # data/ios raw / DentalLabeler label: fdi, label (클래스), instance.  잠깐 쓴 scalar_ 접두어 이름도
    # 읽는다.  label 없이 fdi만 있으면 fdi > 0을 치아로 본다
    fdi, labels, ins = (vert.get(k, vert.get(f"scalar_{k}")) for k in ("fdi", "label", "instance"))
    out.fdi = None if fdi is None else torch.from_numpy(fdi.astype(np.int64))
    out.instances = None if ins is None else torch.from_numpy(ins.astype(np.int64))
    out.labels = (torch.from_numpy(labels.astype(np.int64)) if labels is not None
                  else None if out.fdi is None else (out.fdi > 0).long())

    face = tables.get("face", {})
    idx = next((v for v in face.values() if v.ndim == 2), None)
    out.faces = torch.from_numpy(np.ascontiguousarray(idx, np.int64)) if idx is not None and len(idx) else None
    frgb = _rgb(face)
    out.face_colours = None if frgb is None or out.faces is None else torch.from_numpy(
        frgb.astype(np.uint8) if frgb.dtype.kind in "iu" else np.round(frgb * 255).astype(np.uint8))
    return True


def _rgb(table: dict[str, np.ndarray]) -> np.ndarray | None:
    for names in _RGB_NAMES:
        if all(k in table for k in names):
            return np.stack([table[k] for k in names], 1)
    return None


def _parse_header(text: str) -> tuple[str, list[_Element]]:
    fmt, elements = "", []
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append(_Element(tok[1], int(tok[2])))
        elif tok[0] == "property" and elements:
            if tok[1] == "list":
                elements[-1].props.append((tok[4], _TYPES[tok[2]], _TYPES[tok[3]]))
            else:
                elements[-1].props.append((tok[2], _TYPES[tok[1]]))
    return fmt, elements


def _read_binary(buf: bytes, offset: int, elements: list[_Element],
                 endian: str) -> dict[str, dict[str, np.ndarray]]:
    tables: dict[str, dict[str, np.ndarray]] = {}
    for el in elements:
        lists = [p for p in el.props if len(p) == 3]
        if len(lists) > 1:
            break                     # list가 둘 이상인 element; 그 뒤로는 필요한 게 없다
        # 빠른 길: polygon이 전부 삼각형이면 record 크기가 고정이다
        fields_ = []
        for p in el.props:
            fields_ += [("_n", endian + p[1]), (p[0], endian + p[2], 3)] if len(p) == 3 else [(p[0], endian + p[1])]
        dt = np.dtype(fields_)
        if offset + dt.itemsize * el.count <= len(buf):
            rec = np.frombuffer(buf, dt, el.count, offset)
            if not lists or np.all(rec["_n"] == 3):
                tables[el.name] = {n: rec[n] for n in dt.names if n != "_n"}
                offset += dt.itemsize * el.count
                continue
        if not lists:
            break                     # 파일이 잘렸다
        records = []
        for _ in range(el.count):
            rec_, offset = _binary_record(buf, offset, el, endian)
            records.append(rec_)
        tables[el.name] = _fan(records, el)
    return tables


def _binary_record(buf: bytes, offset: int, el: _Element, endian: str) -> tuple[dict[str, object], int]:
    rec: dict[str, object] = {}
    for p in el.props:
        if len(p) == 3:
            n = int(np.frombuffer(buf, endian + p[1], 1, offset)[0])
            offset += np.dtype(p[1]).itemsize
            rec[p[0]] = np.frombuffer(buf, endian + p[2], n, offset).astype(np.int64)
            offset += np.dtype(p[2]).itemsize * n
        else:
            rec[p[0]] = np.frombuffer(buf, endian + p[1], 1, offset)[0]
            offset += np.dtype(p[1]).itemsize
    return rec, offset


def _read_ascii(body: bytes, elements: list[_Element]) -> dict[str, dict[str, np.ndarray]]:
    lines = body.split(b"\n")
    tables: dict[str, dict[str, np.ndarray]] = {}
    at = 0
    for el in elements:
        rows = lines[at:at + el.count]
        at += el.count
        lists = [p for p in el.props if len(p) == 3]
        if len(lists) > 1:
            break
        # 빠른 길: 줄마다 칸 수가 같고 list가 전부 삼각형이면 한 번에 읽는다
        width = sum(4 if len(p) == 3 else 1 for p in el.props)
        vals = np.fromstring(b" ".join(rows), dtype=np.float64, sep=" ") if el.count else np.empty(0)
        if vals.size == el.count * width:
            vals = vals.reshape(el.count, width)
            table, c = {}, 0
            for p in el.props:
                if len(p) == 3:
                    table[p[0]] = vals[:, c + 1:c + 4].astype(np.int64)
                    ok = bool(np.all(vals[:, c] == 3))
                    c += 4
                else:
                    table[p[0]] = vals[:, c].astype(p[1])
                    c += 1
            if not lists or ok:
                tables[el.name] = table
                continue
        records = []
        for t in (r.split() for r in rows):
            rec, c = {}, 0
            for p in el.props:
                if len(p) == 3:
                    n = int(t[c])
                    rec[p[0]] = np.array(t[c + 1:c + 1 + n], dtype=np.int64)
                    c += 1 + n
                else:
                    rec[p[0]] = float(t[c])
                    c += 1
            records.append(rec)
        tables[el.name] = _fan(records, el)
    return tables


def _fan(records: list[dict[str, object]], el: _Element) -> dict[str, np.ndarray]:
    """polygon을 fan으로 삼각형으로 나누고, scalar 속성은 삼각형마다 복사한다."""
    cols: dict[str, list] = {p[0]: [] for p in el.props}
    lname = next(p[0] for p in el.props if len(p) == 3)
    for rec in records:
        idx = rec[lname]
        for j in range(1, len(idx) - 1):
            cols[lname].append((idx[0], idx[j], idx[j + 1]))
            for p in el.props:
                if len(p) == 2:
                    cols[p[0]].append(rec[p[0]])
    return {p[0]: (np.asarray(cols[p[0]], np.int64).reshape(-1, 3) if len(p) == 3
                   else np.asarray(cols[p[0]], p[1])) for p in el.props}
