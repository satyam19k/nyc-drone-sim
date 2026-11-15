"""Minimal glTF binary reader for the model sets used to populate the scene. CPU only.

Reads positions, triangle indices, materials and selected custom vertex attributes of one
level-of-detail node. Handles the two encodings the model sets use: quantised vertex data
(KHR_mesh_quantization) and buffer compression (EXT_meshopt_compression: vertex codec
version 0, index codec versions 0 and 1, no filters). Textures are not read.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_NORM = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}


# ---------------------------------------------------------------------------
# Buffer decompression
# ---------------------------------------------------------------------------
def _decode_bytes(data: bytes, pos: int, size: int) -> tuple[np.ndarray, int]:
    """One byte stream of `size` bytes (a multiple of 16), stored as 16-byte groups."""
    groups = size // 16
    header = pos
    pos += (groups + 3) // 4
    out = np.zeros(size, dtype=np.uint8)
    for g in range(groups):
        bits = (data[header + g // 4] >> ((g % 4) * 2)) & 3
        if bits == 0:
            continue
        o = g * 16
        if bits == 3:
            out[o:o + 16] = np.frombuffer(data, np.uint8, 16, pos)
            pos += 16
            continue
        per, width, sentinel = (4, 2, 3) if bits == 1 else (2, 4, 15)
        packed = pos
        pos += 16 // per
        for i in range(16):
            v = (data[packed + i // per] >> ((per - 1 - i % per) * width)) & sentinel
            if v == sentinel:
                v = data[pos]
                pos += 1
            out[o + i] = v
    return out, pos


def decode_vertex_buffer(data: bytes, count: int, stride: int) -> np.ndarray:
    if data[0] != 0xA0:
        raise ValueError(f"unsupported vertex codec header {data[0]:#x}")
    block_max = min((8192 // stride) & ~15, 256)
    last = np.frombuffer(data, np.uint8, stride, len(data) - stride).copy()
    out = np.zeros((count, stride), dtype=np.uint8)
    pos, done = 1, 0
    while done < count:
        n = min(block_max, count - done)
        padded = (n + 15) & ~15
        for k in range(stride):
            buf, pos = _decode_bytes(data, pos, padded)
            d = buf[:n].astype(np.int32)
            d = np.where(d & 1, ~(d >> 1), d >> 1)                 # undo zigzag
            out[done:done + n, k] = (int(last[k]) + np.cumsum(d)) & 0xFF
        last = out[done + n - 1].copy()
        done += n
    return out


def decode_index_buffer(data: bytes, count: int) -> np.ndarray:
    if data[0] & 0xF0 != 0xE0 or (data[0] & 0x0F) > 1:
        raise ValueError(f"unsupported index codec header {data[0]:#x}")
    fecmax = 13 if (data[0] & 0x0F) >= 1 else 15
    edges = [(0xFFFFFFFF, 0xFFFFFFFF)] * 16
    verts = [0xFFFFFFFF] * 16
    eo = vo = 0
    nxt = last = 0
    code = 1
    pos = 1 + count // 3
    aux_table = len(data) - 16
    out = np.zeros(count, dtype=np.uint32)

    def read_index():
        nonlocal pos, last
        v, shift = 0, 0
        while True:
            b = data[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                break
        last = (last + ((v >> 1) ^ -(v & 1))) & 0xFFFFFFFF
        return last

    for i in range(0, count, 3):
        tri = data[code]
        code += 1
        if tri < 0xF0:
            a, b = edges[(eo - 1 - (tri >> 4)) & 15]
            fec = tri & 15
            if fec < fecmax:
                c = nxt if fec == 0 else verts[(vo - 1 - fec) & 15]
                if fec == 0:
                    nxt += 1
                    verts[vo] = c
                    vo = (vo + 1) & 15
            else:
                if fec != 15:
                    last = c = (last + (fec - (fec ^ 3))) & 0xFFFFFFFF
                else:
                    c = read_index()
                verts[vo] = c
                vo = (vo + 1) & 15
            edges[eo] = (c, b)
            eo = (eo + 1) & 15
            edges[eo] = (a, c)
            eo = (eo + 1) & 15
        else:
            if tri < 0xFE:
                aux = data[aux_table + (tri & 15)]
                feb, fec = aux >> 4, aux & 15
                a = nxt
                nxt += 1
                b = nxt if feb == 0 else verts[(vo - feb) & 15]
                nxt += feb == 0
                c = nxt if fec == 0 else verts[(vo - fec) & 15]
                nxt += fec == 0
                verts[vo] = a
                vo = (vo + 1) & 15
                verts[vo] = b
                vo = (vo + (feb == 0)) & 15
                verts[vo] = c
                vo = (vo + (fec == 0)) & 15
            else:
                aux = data[pos]
                pos += 1
                fea = 0 if tri == 0xFE else 15
                feb, fec = aux >> 4, aux & 15
                if aux == 0:
                    nxt = 0        # restart of the sequential numbering, e.g. at a new primitive
                a = 0
                if fea == 0:
                    a = nxt
                    nxt += 1
                if feb == 0:
                    b = nxt
                    nxt += 1
                else:
                    b = verts[(vo - feb) & 15]
                if fec == 0:
                    c = nxt
                    nxt += 1
                else:
                    c = verts[(vo - fec) & 15]
                if fea == 15:
                    a = read_index()
                if feb == 15:
                    b = read_index()
                if fec == 15:
                    c = read_index()
                verts[vo] = a
                vo = (vo + 1) & 15
                verts[vo] = b
                vo = (vo + (feb in (0, 15))) & 15
                verts[vo] = c
                vo = (vo + (fec in (0, 15))) & 15
            edges[eo] = (b, a)
            eo = (eo + 1) & 15
            edges[eo] = (c, b)
            eo = (eo + 1) & 15
            edges[eo] = (a, c)
            eo = (eo + 1) & 15
        out[i], out[i + 1], out[i + 2] = a, b, c
    return out


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class Glb:
    def __init__(self, path: Path):
        raw = Path(path).read_bytes()
        if raw[:4] != b"glTF":
            raise ValueError(f"{path} is not a glTF binary")
        n = struct.unpack_from("<I", raw, 12)[0]
        self.json = json.loads(raw[20:20 + n])
        o = 20 + n
        self.bin = raw[o + 8:o + 8 + struct.unpack_from("<I", raw, o)[0]] if o < len(raw) else b""
        self._views: dict[int, bytes] = {}

    def view(self, i: int) -> bytes:
        if i not in self._views:
            bv = self.json["bufferViews"][i]
            ext = bv.get("extensions", {}).get("EXT_meshopt_compression")
            if ext is None:
                o = bv.get("byteOffset", 0)
                self._views[i] = self.bin[o:o + bv["byteLength"]]
            else:
                if ext.get("filter", "NONE") != "NONE":
                    raise ValueError(f"unsupported compression filter {ext['filter']}")
                o = ext.get("byteOffset", 0)
                src = self.bin[o:o + ext["byteLength"]]
                if ext["mode"] == "ATTRIBUTES":
                    self._views[i] = decode_vertex_buffer(src, ext["count"], ext["byteStride"]).tobytes()
                elif ext["mode"] == "TRIANGLES":
                    idx = decode_index_buffer(src, ext["count"])
                    self._views[i] = idx.astype({2: np.uint16, 4: np.uint32}[ext["byteStride"]]).tobytes()
                else:
                    raise ValueError(f"unsupported compression mode {ext['mode']}")
        return self._views[i]

    def accessor(self, i: int) -> np.ndarray:
        a = self.json["accessors"][i]
        dt, w = _COMPONENT[a["componentType"]], _WIDTH[a["type"]]
        bv = self.json["bufferViews"][a["bufferView"]]
        stride = bv.get("byteStride") or bv.get("extensions", {}).get("EXT_meshopt_compression", {}).get("byteStride")
        item = np.dtype(dt).itemsize * w
        data = self.view(a["bufferView"])
        o = a.get("byteOffset", 0)
        if stride and stride != item:
            rows = np.frombuffer(data, np.uint8, offset=o, count=(a["count"] - 1) * stride + item)
            idx = np.arange(a["count"])[:, None] * stride + np.arange(item)[None, :]
            arr = rows[idx].copy().view(dt).reshape(a["count"], w)
        else:
            arr = np.frombuffer(data, dt, a["count"] * w, o).reshape(a["count"], w)
        if a.get("normalized") and dt in _NORM:
            arr = np.maximum(arr.astype(np.float64) / _NORM[dt], -1.0) if dt in (np.int8, np.int16) else arr / _NORM[dt]
        return arr

    def node_matrix(self, i: int) -> np.ndarray:
        n = self.json["nodes"][i]
        if "matrix" in n:
            return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        x, y, z, w = n.get("rotation", [0, 0, 0, 1])
        m[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m[:3, :3] *= np.array(n.get("scale", [1, 1, 1]), dtype=np.float64)[None, :]
        m[:3, 3] = n.get("translation", [0, 0, 0])
        return m

    def primitives(self, lod: int = 0, extra: tuple[str, ...] = ()) -> list[dict]:
        """Triangle primitives of the `LOD<lod>` subtree (or the whole scene if no LOD nodes exist).

        Positions are returned in the file's frame (metres, +Y up), node transforms applied.
        """
        nodes = self.json["nodes"]
        names = [n.get("name", "") for n in nodes]
        has_lod = any(nm.startswith("LOD") for nm in names)
        out = []

        def walk(i: int, parent: np.ndarray, active: bool):
            m = parent @ self.node_matrix(i)
            nm = names[i]
            if has_lod and nm.startswith("LOD"):
                active = nm == f"LOD{lod}"
            elif not has_lod:
                active = True
            if active and "mesh" in nodes[i]:
                for pr in self.json["meshes"][nodes[i]["mesh"]]["primitives"]:
                    if pr.get("mode", 4) != 4:
                        continue
                    p = np.asarray(self.accessor(pr["attributes"]["POSITION"]), dtype=np.float64)
                    p = p @ m[:3, :3].T + m[:3, 3]
                    tri = (np.asarray(self.accessor(pr["indices"])).reshape(-1, 3).astype(np.int64)
                           if "indices" in pr else np.arange(len(p)).reshape(-1, 3))
                    mat = self.json["materials"][pr["material"]] if "material" in pr else {}
                    out.append({"points": p, "triangles": tri, "material": mat,
                                **{k: np.asarray(self.accessor(pr["attributes"][k])) for k in extra
                                   if k in pr["attributes"]}})
            for c in nodes[i].get("children", []):
                walk(c, m, active)

        scene = self.json["scenes"][self.json.get("scene", 0)]
        for r in scene["nodes"]:
            walk(r, np.eye(4), False)
        return out
