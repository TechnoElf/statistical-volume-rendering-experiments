"""
Pure-Python reader/writer for uncompressed single-grid VDB files.
Tree topology: 5-4-3 (Node5 → Node4 → Leaf), voxel values stored as float32.
Only numpy is used beyond the standard library.

File format reference:
  https://jangafx.com/insights/vdb-a-deep-dive
  https://github.com/jangafx/simple-vdb-writer
"""

from __future__ import annotations

import io
import struct
import uuid as _uuid_mod
from typing import Iterator, Optional

import numpy as np

# ── file-level constants ──────────────────────────────────────────────────────

_MAGIC        = b'\x20\x42\x44\x56\x00\x00\x00\x00'
_FILE_VERSION = 224
_LIB_MAJOR    = 8
_LIB_MINOR    = 1
_GRID_TYPE    = "Tree_float_5_4_3"

# Value written before each uncompressed data block to signal no compression.
_NO_COMPRESS  = 6

# ── 5-4-3 tree geometry ───────────────────────────────────────────────────────
#
#  Node5  log2dim=5  32^3=32768 child slots  spans 32*128 = 4096 voxels/axis
#  Node4  log2dim=4  16^3=4096  child slots  spans 16*8   = 128  voxels/axis
#  Leaf   log2dim=3  8^3=512    voxel slots  spans 8             voxels/axis

_SPAN5 = 4096   # voxels per Node5 axis
_SPAN4 = 128    # voxels per Node4 axis
_SPAN3 = 8      # voxels per Leaf axis

_MW5 = 512      # uint64 mask words per Node5 (32^3 / 64)
_MW4 = 64       # uint64 mask words per Node4 (16^3 / 64)
_MW3 = 8        # uint64 mask words per Leaf  (8^3  / 64)

_NC5 = 32768    # total child slots in Node5
_NC4 = 4096     # total child slots in Node4
_NV3 = 512      # total voxel slots in Leaf

# ── bitmask helpers ───────────────────────────────────────────────────────────

def _mask_set(mask: np.ndarray, idx: int) -> None:
    w, b = idx >> 6, idx & 63
    mask[w] = np.uint64(int(mask[w]) | (1 << b))

def _mask_get(mask: np.ndarray, idx: int) -> bool:
    return bool(int(mask[idx >> 6]) & (1 << (idx & 63)))

def _iter_set(mask: np.ndarray) -> Iterator[int]:
    """Yield the position of every set bit in a uint64 mask array."""
    for wi in range(len(mask)):
        w = int(mask[wi])
        while w:
            bit = (w & -w).bit_length() - 1
            yield wi * 64 + bit
            w &= w - 1

# ── coordinate helpers ────────────────────────────────────────────────────────

def _node5_origin(x: int, y: int, z: int) -> tuple[int, int, int]:
    s = _SPAN5
    return x // s * s, y // s * s, z // s * s   # floor-division handles negatives

def _bit4(lx: int, ly: int, lz: int) -> int:
    """Slot of Node4 within Node5 (local coords in [0, SPAN5))."""
    return (lz >> 7) | ((ly >> 7) << 5) | ((lx >> 7) << 10)

def _bit3(lx: int, ly: int, lz: int) -> int:
    """Slot of Leaf within Node4 (local coords in [0, SPAN4))."""
    return (lz >> 3) | ((ly >> 3) << 4) | ((lx >> 3) << 8)

def _bit0(lx: int, ly: int, lz: int) -> int:
    """Voxel index within Leaf (local coords in [0, SPAN3))."""
    return lz | (ly << 3) | (lx << 6)

def _decode4(slot: int) -> tuple[int, int, int]:
    return (slot >> 10) & 31, (slot >> 5) & 31, slot & 31

def _decode3(slot: int) -> tuple[int, int, int]:
    return (slot >> 8) & 15, (slot >> 4) & 15, slot & 15

def _decode0(idx: int) -> tuple[int, int, int]:
    return (idx >> 6) & 7, (idx >> 3) & 7, idx & 7

# ── tree nodes ────────────────────────────────────────────────────────────────

class _Leaf:
    __slots__ = ("value_mask", "data")

    def __init__(self) -> None:
        self.value_mask = np.zeros(_MW3, np.uint64)
        self.data       = np.zeros(_NV3, np.float32)

    def set(self, lx: int, ly: int, lz: int, v: float) -> None:
        idx = _bit0(lx, ly, lz)
        _mask_set(self.value_mask, idx)
        self.data[idx] = v

    def get(self, lx: int, ly: int, lz: int, bg: float) -> float:
        idx = _bit0(lx, ly, lz)
        return float(self.data[idx]) if _mask_get(self.value_mask, idx) else bg


class _Node4:
    __slots__ = ("child_mask", "value_mask", "children")

    def __init__(self) -> None:
        self.child_mask  = np.zeros(_MW4, np.uint64)
        self.value_mask  = np.zeros(_MW4, np.uint64)
        self.children: dict[int, _Leaf] = {}

    def get_or_create_leaf(self, slot: int) -> _Leaf:
        if slot not in self.children:
            self.children[slot] = _Leaf()
            _mask_set(self.child_mask, slot)
        return self.children[slot]


class _Node5:
    __slots__ = ("origin", "child_mask", "value_mask", "children")

    def __init__(self, origin: tuple[int, int, int]) -> None:
        self.origin      = origin
        self.child_mask  = np.zeros(_MW5, np.uint64)
        self.value_mask  = np.zeros(_MW5, np.uint64)
        self.children: dict[int, _Node4] = {}

    def get_or_create_node4(self, slot: int) -> _Node4:
        if slot not in self.children:
            self.children[slot] = _Node4()
            _mask_set(self.child_mask, slot)
        return self.children[slot]

# ── public API ────────────────────────────────────────────────────────────────

class VDBGrid:
    """
    Sparse float32 VDB grid using a 5-4-3 tree.

    Basic usage::

        g = VDBGrid("density")
        g.set_voxel(0, 0, 0, 1.0)
        write_vdb("out.vdb", g)

        g2 = read_vdb("out.vdb")
        print(g2.get_voxel(0, 0, 0))   # 1.0
    """

    def __init__(
        self,
        name: str = "density",
        background: float = 0.0,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        self.name        = name
        self.background  = float(background)
        # Row-major 4×4 affine matrix mapping index space to world space.
        self.transform: np.ndarray = (
            np.eye(4, dtype=np.float64)
            if transform is None
            else np.asarray(transform, dtype=np.float64)
        )
        self._node5s: dict[tuple[int, int, int], _Node5] = {}

    # ── voxel access ──────────────────────────────────────────────────────────

    def set_voxel(self, x: int, y: int, z: int, value: float) -> None:
        """Activate the voxel at (x, y, z) and assign value."""
        origin = _node5_origin(x, y, z)
        n5 = self._node5s.get(origin)
        if n5 is None:
            n5 = _Node5(origin)
            self._node5s[origin] = n5

        lx5, ly5, lz5 = x - origin[0], y - origin[1], z - origin[2]

        n4 = n5.get_or_create_node4(_bit4(lx5, ly5, lz5))
        lx4, ly4, lz4 = lx5 & (_SPAN4 - 1), ly5 & (_SPAN4 - 1), lz5 & (_SPAN4 - 1)

        leaf = n4.get_or_create_leaf(_bit3(lx4, ly4, lz4))
        leaf.set(lx5 & (_SPAN3 - 1), ly5 & (_SPAN3 - 1), lz5 & (_SPAN3 - 1), value)

    def get_voxel(self, x: int, y: int, z: int) -> float:
        """Return the value at (x, y, z), or background if inactive."""
        origin = _node5_origin(x, y, z)
        n5 = self._node5s.get(origin)
        if n5 is None:
            return self.background

        lx5, ly5, lz5 = x - origin[0], y - origin[1], z - origin[2]
        slot4 = _bit4(lx5, ly5, lz5)
        if not _mask_get(n5.child_mask, slot4):
            return self.background

        n4 = n5.children[slot4]
        lx4, ly4, lz4 = lx5 & (_SPAN4 - 1), ly5 & (_SPAN4 - 1), lz5 & (_SPAN4 - 1)
        slot3 = _bit3(lx4, ly4, lz4)
        if not _mask_get(n4.child_mask, slot3):
            return self.background

        leaf = n4.children[slot3]
        return leaf.get(lx5 & (_SPAN3 - 1), ly5 & (_SPAN3 - 1), lz5 & (_SPAN3 - 1), self.background)

    # ── bulk helpers ──────────────────────────────────────────────────────────

    def set_dense(self, origin: tuple[int, int, int], array: np.ndarray) -> None:
        """
        Load a dense (X, Y, Z) float32 array into the grid.
        Only voxels that differ from background are activated.
        origin is the (x, y, z) world coordinate of array[0, 0, 0].
        """
        array = np.asarray(array, dtype=np.float32)
        bg    = np.float32(self.background)
        ox, oy, oz = origin
        for xi, yi, zi in zip(*np.nonzero(array != bg)):
            self.set_voxel(int(ox + xi), int(oy + yi), int(oz + zi),
                           float(array[xi, yi, zi]))

    def iter_active_voxels(self) -> Iterator[tuple[int, int, int, float]]:
        """Yield (x, y, z, value) for every active voxel."""
        for (n5x, n5y, n5z), n5 in self._node5s.items():
            for slot4 in _iter_set(n5.child_mask):
                ix5, iy5, iz5 = _decode4(slot4)
                n4 = n5.children[slot4]
                for slot3 in _iter_set(n4.child_mask):
                    ix4, iy4, iz4 = _decode3(slot3)
                    leaf = n4.children[slot3]
                    for vi in _iter_set(leaf.value_mask):
                        ixv, iyv, izv = _decode0(vi)
                        x = n5x + ix5 * _SPAN4 + ix4 * _SPAN3 + ixv
                        y = n5y + iy5 * _SPAN4 + iy4 * _SPAN3 + iyv
                        z = n5z + iz5 * _SPAN4 + iz4 * _SPAN3 + izv
                        yield x, y, z, float(leaf.data[vi])

    def to_dense(
        self,
        origin: tuple[int, int, int],
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Extract a dense (X, Y, Z) float32 array for the voxel region
        [origin, origin + shape).  Inactive voxels get background value.
        """
        out = np.full(shape, self.background, dtype=np.float32)
        ox, oy, oz = origin
        sx, sy, sz = shape
        for x, y, z, v in self.iter_active_voxels():
            lx, ly, lz = x - ox, y - oy, z - oz
            if 0 <= lx < sx and 0 <= ly < sy and 0 <= lz < sz:
                out[lx, ly, lz] = v
        return out

    def active_count(self) -> int:
        """Return the total number of active voxels."""
        count = 0
        for n5 in self._node5s.values():
            for slot4 in _iter_set(n5.child_mask):
                n4 = n5.children[slot4]
                for slot3 in _iter_set(n4.child_mask):
                    leaf = n4.children[slot3]
                    count += sum(bin(int(w)).count('1') for w in leaf.value_mask)
        return count

# ── serialization helpers ─────────────────────────────────────────────────────

def _pack_name(s: str) -> bytes:
    b = s.encode()
    return struct.pack('<I', len(b)) + b

def _pack_meta_string(name: str, value: str) -> bytes:
    return _pack_name(name) + _pack_name("string") + _pack_name(value)

def _read_name(buf: io.RawIOBase) -> str:
    (n,) = struct.unpack('<I', buf.read(4))
    return buf.read(n).decode()

def _skip_meta(buf: io.RawIOBase) -> None:
    _read_name(buf)                          # entry name
    _read_name(buf)                          # type name
    (n,) = struct.unpack('<I', buf.read(4))  # value byte count
    buf.read(n)

def _read_meta_entry(buf: io.RawIOBase) -> tuple[str, str, bytes]:
    name  = _read_name(buf)
    type_ = _read_name(buf)
    (n,)  = struct.unpack('<I', buf.read(4))
    data  = buf.read(n)
    return name, type_, data

# ── writer ────────────────────────────────────────────────────────────────────

def write_vdb(path: str, grid: VDBGrid) -> None:
    """Write grid to path as an uncompressed VDB file."""
    buf = io.BytesIO()

    # File header
    buf.write(_MAGIC)
    buf.write(struct.pack('<III', _FILE_VERSION, _LIB_MAJOR, _LIB_MINOR))
    buf.write(struct.pack('<B', 0))                          # no grid-offset table
    buf.write(str(_uuid_mod.uuid4()).encode())               # 36-byte UUID, no length prefix
    buf.write(struct.pack('<II', 0, 1))                      # 0 file-meta entries, 1 grid

    _write_grid(buf, grid)

    with open(path, 'wb') as f:
        f.write(buf.getvalue())


def _write_grid(buf: io.BytesIO, grid: VDBGrid) -> None:
    buf.write(_pack_name(grid.name))
    buf.write(_pack_name(_GRID_TYPE))
    buf.write(struct.pack('<I', 0))                         # no instance parent

    # Three u64 offsets.  First points just past these three fields.
    pos = buf.tell()
    buf.write(struct.pack('<QQQ', pos + 24, 0, 0))

    buf.write(struct.pack('<I', 0))                         # no compression

    _write_metadata(buf, grid)
    _write_transform(buf, grid.transform)
    _write_tree(buf, grid)


def _write_metadata(buf: io.BytesIO, grid: VDBGrid) -> None:
    buf.write(struct.pack('<I', 3))                         # 3 metadata entries
    buf.write(_pack_meta_string("class", "unknown"))
    buf.write(_pack_meta_string("file_compression", "none"))
    buf.write(_pack_meta_string("name", grid.name))


def _write_transform(buf: io.BytesIO, m: np.ndarray) -> None:
    buf.write(_pack_name("AffineMap"))
    # 4x4 column-major: for each column, write 3 doubles from rows 0-2,
    # then the homogeneous component (0 for linear columns, 1 for translation).
    for col in range(3):
        for row in range(3):
            buf.write(struct.pack('<d', float(m[row, col])))
        buf.write(struct.pack('<d', 0.0))
    for row in range(3):
        buf.write(struct.pack('<d', float(m[row, 3])))
    buf.write(struct.pack('<d', 1.0))


_ZEROS_NC5 = np.zeros(_NC5, np.float32).tobytes()
_ZEROS_NC4 = np.zeros(_NC4, np.float32).tobytes()


def _write_tree(buf: io.BytesIO, grid: VDBGrid) -> None:
    node5_list = list(grid._node5s.values())

    buf.write(struct.pack('<I', 1))                         # buffer count
    buf.write(struct.pack('<f', grid.background))
    buf.write(struct.pack('<I', 0))                         # no root tiles
    buf.write(struct.pack('<I', len(node5_list)))

    # ── topology pass ─────────────────────────────────────────────────────────
    # Node5 headers, then Node4 headers, then Leaf value-masks.
    for n5 in node5_list:
        _write_node5_header(buf, n5)
        for slot4 in _iter_set(n5.child_mask):
            n4 = n5.children[slot4]
            _write_node4_header(buf, n4)
            for slot3 in _iter_set(n4.child_mask):
                buf.write(n4.children[slot3].value_mask.tobytes())

    # ── data pass ─────────────────────────────────────────────────────────────
    # Same traversal order: Leaf value-mask, compression byte, then voxel data.
    for n5 in node5_list:
        for slot4 in _iter_set(n5.child_mask):
            n4 = n5.children[slot4]
            for slot3 in _iter_set(n4.child_mask):
                leaf = n4.children[slot3]
                buf.write(leaf.value_mask.tobytes())
                buf.write(struct.pack('<B', _NO_COMPRESS))
                buf.write(leaf.data.tobytes())


def _write_node5_header(buf: io.BytesIO, n5: _Node5) -> None:
    ox, oy, oz = n5.origin
    buf.write(struct.pack('<iii', ox, oy, oz))
    buf.write(n5.child_mask.tobytes())
    buf.write(n5.value_mask.tobytes())
    buf.write(struct.pack('<B', _NO_COMPRESS))
    buf.write(_ZEROS_NC5)                                   # tile values (all background)


def _write_node4_header(buf: io.BytesIO, n4: _Node4) -> None:
    buf.write(n4.child_mask.tobytes())
    buf.write(n4.value_mask.tobytes())
    buf.write(struct.pack('<B', _NO_COMPRESS))
    buf.write(_ZEROS_NC4)                                   # tile values (all background)

# ── reader ────────────────────────────────────────────────────────────────────

def read_vdb(path: str) -> VDBGrid:
    """Read a single-grid VDB file and return a VDBGrid."""
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    return _read_file(buf)


def _read_file(buf: io.BytesIO) -> VDBGrid:
    magic = buf.read(8)
    if magic != _MAGIC:
        raise ValueError(f"Not a VDB file (magic bytes: {magic!r})")

    (file_ver,)    = struct.unpack('<I', buf.read(4))
    (lib_major,)   = struct.unpack('<I', buf.read(4))
    (lib_minor,)   = struct.unpack('<I', buf.read(4))
    (has_offsets,) = struct.unpack('<B', buf.read(1))
    _uuid          = buf.read(36).decode()

    (meta_count,) = struct.unpack('<I', buf.read(4))
    for _ in range(meta_count):
        _skip_meta(buf)

    (grid_count,) = struct.unpack('<I', buf.read(4))
    if grid_count == 0:
        raise ValueError("VDB file contains no grids")
    if grid_count > 1:
        raise ValueError(
            f"Multi-grid VDB files are not supported (found {grid_count} grids)"
        )

    return _read_grid(buf)


def _read_transform(buf: io.BytesIO) -> np.ndarray:
    map_type = _read_name(buf)
    if map_type != "AffineMap":
        raise ValueError(f"Unsupported transform type: {map_type!r}")
    vals = struct.unpack('<16d', buf.read(128))
    m = np.eye(4, dtype=np.float64)
    for col in range(4):
        for row in range(3):
            m[row, col] = vals[col * 4 + row]
    # vals[col*4 + 3] are homogeneous components (0 or 1), already correct in eye(4)
    return m


def _read_grid(buf: io.BytesIO) -> VDBGrid:
    name      = _read_name(buf)
    grid_type = _read_name(buf)
    (_inst,)  = struct.unpack('<I', buf.read(4))
    buf.read(24)                                            # three u64 offsets

    (_compress,) = struct.unpack('<I', buf.read(4))

    half = grid_type.endswith("HalfFloat")
    (meta_count,) = struct.unpack('<I', buf.read(4))
    for _ in range(meta_count):
        mname, mtype, mdata = _read_meta_entry(buf)
        if mname == "is_saved_as_half_float" and mtype == "bool":
            half = bool(mdata[0])

    transform = _read_transform(buf)
    grid = VDBGrid(name=name, transform=transform)
    _read_tree(buf, grid, half)
    return grid


def _read_tree(buf: io.BytesIO, grid: VDBGrid, half: bool) -> None:
    (_one,)     = struct.unpack('<I', buf.read(4))
    (bg_raw,)   = struct.unpack('<f', buf.read(4))
    (_tiles,)   = struct.unpack('<I', buf.read(4))
    (n5_count,) = struct.unpack('<I', buf.read(4))

    vbytes = 2 if half else 4
    vdtype = np.float16 if half else np.float32

    grid.background = float(bg_raw)

    # We collect node references in traversal order so the data pass
    # can re-use the same iteration without re-reading the masks.
    all_n5: list[tuple[_Node5, list[tuple[_Node4, list[tuple[int, _Leaf]]]]]] = []

    # ── topology pass ─────────────────────────────────────────────────────────
    for _ in range(n5_count):
        ox, oy, oz = struct.unpack('<iii', buf.read(12))
        child_mask5 = np.frombuffer(buf.read(_MW5 * 8), np.uint64).copy()
        value_mask5 = np.frombuffer(buf.read(_MW5 * 8), np.uint64).copy()
        (_c5,)      = struct.unpack('<B', buf.read(1))
        buf.read(_NC5 * vbytes)                             # skip tile values

        n5 = _Node5((ox, oy, oz))
        n5.child_mask[:] = child_mask5
        n5.value_mask[:] = value_mask5

        n4_list: list[tuple[_Node4, list[tuple[int, _Leaf]]]] = []

        for slot4 in _iter_set(child_mask5):
            child_mask4 = np.frombuffer(buf.read(_MW4 * 8), np.uint64).copy()
            value_mask4 = np.frombuffer(buf.read(_MW4 * 8), np.uint64).copy()
            (_c4,)      = struct.unpack('<B', buf.read(1))
            buf.read(_NC4 * vbytes)                         # skip tile values

            n4 = _Node4()
            n4.child_mask[:] = child_mask4
            n4.value_mask[:] = value_mask4

            leaf_list: list[tuple[int, _Leaf]] = []

            for slot3 in _iter_set(child_mask4):
                vm3  = np.frombuffer(buf.read(_MW3 * 8), np.uint64).copy()
                leaf = _Leaf()
                leaf.value_mask[:] = vm3
                n4.children[slot3] = leaf
                leaf_list.append((slot3, leaf))

            n5.children[slot4] = n4
            n4_list.append((n4, leaf_list))

        grid._node5s[(ox, oy, oz)] = n5
        all_n5.append((n5, n4_list))

    # ── data pass ─────────────────────────────────────────────────────────────
    for _n5, n4_list in all_n5:
        for _n4, leaf_list in n4_list:
            for _slot3, leaf in leaf_list:
                vm3  = np.frombuffer(buf.read(_MW3 * 8), np.uint64).copy()
                leaf.value_mask[:] = vm3
                (_c,) = struct.unpack('<B', buf.read(1))
                raw   = np.frombuffer(buf.read(_NV3 * vbytes), vdtype)
                leaf.data[:] = raw.astype(np.float32)
