"""CPU XIM decoding based on TIGRE's MATLAB ReadXim/mexReadXim readers.

Copyright (c) 2015, Fredrik. See ReadXim_license.txt for the original BSD
notice. MATLAB integration and file conventions: Yi Du, 2017-2021.
Images here retain file row order; VarianDataLoader applies MATLAB's flip.
"""
from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np


@dataclass
class XimImage:
    pixels: np.ndarray
    properties: dict
    width: int
    height: int
    version: int


class _Reader:
    def __init__(self, stream, length):
        self.stream, self.length = stream, length

    def read(self, count):
        if count < 0 or count > self.length-self.stream.tell():
            raise ValueError("Truncated or invalid XIM buffer length.")
        value = self.stream.read(count)
        if len(value) != count:
            raise ValueError("Truncated XIM file.")
        return value

    def integer(self):
        return struct.unpack("<i", self.read(4))[0]


def _decompress(buffer, lookup, width, height):
    count = width*height
    seeds = min(width+1, count)
    if len(buffer) < seeds*4:
        raise ValueError("Truncated XIM predictor seeds.")
    initial = np.frombuffer(buffer, dtype="<i4", count=seeds).astype(np.int64)
    remaining = count-seeds
    if not remaining:
        return initial.astype(np.int32)
    codes = ((np.frombuffer(lookup, dtype=np.uint8)[:, None]
              >> np.array([0, 2, 4, 6])) & 3).ravel()[:remaining]
    if len(codes) != remaining:
        raise ValueError("XIM lookup table is too short.")
    sizes = np.where(codes == 0, 1, np.where(codes == 1, 2, 4))
    offsets = seeds*4 + np.r_[0, np.cumsum(sizes[:-1])]
    if seeds*4+int(sizes.sum()) != len(buffer):
        raise ValueError("XIM compressed buffer size does not match lookup table.")
    raw = np.frombuffer(buffer, dtype=np.uint8)
    residuals = np.empty(remaining, dtype=np.int64)
    for size in (1, 2, 4):
        selected = sizes == size
        positions = offsets[selected]
        unsigned = np.zeros(len(positions), dtype=np.uint32)
        for byte in range(size):
            unsigned |= raw[positions+byte].astype(np.uint32) << (8*byte)
        signed = unsigned.astype(f"<u{size}").view(f"<i{size}")
        residuals[selected] = signed
    # Predictor p[i]=d[i]+p[i-1]+p[i-w]-p[i-w-1]. Integrate horizontal
    # differences down each w-spaced lane, then integrate along the stream.
    increments = np.zeros(count, dtype=np.int64)
    increments[:width] = np.diff(initial)
    increments[width:count-1] = residuals
    gradients = np.cumsum(increments.reshape(height, width), axis=0).ravel()[:count-1]
    decoded = initial[0] + np.r_[0, np.cumsum(gradients)]
    if np.any(decoded < np.iinfo(np.int32).min) or np.any(decoded > np.iinfo(np.int32).max):
        raise ValueError("XIM decoded pixels exceed int32 range.")
    return decoded.astype(np.int32)


def read_xim(path, *, max_pixels=64_000_000):
    """Read compressed/uncompressed little-endian XIM pixels and properties.

    Supports property types 0 (int), 1 (double), 2 (text), 4 (double array),
    and 5 (int array). Truncated buffers and unsupported formats raise errors.
    No CUDA, third-party XIM package, or unpickling is used.
    """
    path = Path(path)
    with path.open("rb") as stream:
        reader = _Reader(stream, path.stat().st_size)
        if not reader.read(8).startswith(b"VMS.XI"):
            raise ValueError(f"Not a supported XIM file: {path}")
        version = reader.integer()
        width, height, bits, bpp, compression = [reader.integer() for _ in range(5)]
        count = width*height
        if width < 0 or height < 0 or count > max_pixels or bpp not in (1, 2, 4) or bits != 8*bpp:
            raise ValueError("Invalid XIM image dimensions or pixel type.")
        if compression == 1:
            lookup = reader.read(reader.integer())
            buffer = reader.read(reader.integer())
            size = reader.integer()
            if size not in (count*bpp, count*4):
                raise ValueError("Invalid XIM uncompressed buffer size.")
            pixels = _decompress(buffer, lookup, width, height)
        elif compression == 0:
            size = reader.integer()
            if size != count*bpp:
                raise ValueError("XIM buffer size does not match dimensions.")
            dtype = "<i4" if bpp == 4 else f"<u{bpp}"
            pixels = np.frombuffer(reader.read(size), dtype=dtype).astype(np.int32)
        else:
            raise NotImplementedError(f"XIM compression {compression} is unsupported.")
        bins = reader.integer()
        if bins < 0:
            raise ValueError("Invalid XIM histogram size.")
        reader.read(4*bins)
        property_count = reader.integer()
        if property_count < 0 or property_count > (reader.length-stream.tell())//8:
            raise ValueError("Invalid XIM property count.")
        properties = {}
        for _ in range(property_count):
            name = reader.read(reader.integer()).decode("utf-8", errors="replace")
            kind = reader.integer()
            if kind == 0:
                value = reader.integer()
            elif kind == 1:
                value = struct.unpack("<d", reader.read(8))[0]
            elif kind == 2:
                value = reader.read(reader.integer()).decode("utf-8", errors="replace")
            elif kind in (4, 5):
                size = reader.integer()
                itemsize = 8 if kind == 4 else 4
                if size % itemsize:
                    raise ValueError("Invalid XIM property array size.")
                value = np.frombuffer(reader.read(size), dtype="<f8" if kind == 4 else "<i4").copy()
            else:
                raise NotImplementedError(f"XIM property type {kind} is unsupported.")
            if name in properties:
                raise ValueError(f"Duplicate XIM property: {name}")
            properties[name] = value
    return XimImage(pixels.reshape(height, width), properties, width, height, version)
