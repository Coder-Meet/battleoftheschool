"""Shared NIfTI loading that tolerates gzip-under-.nii files and slightly non-orthonormal headers."""

import gzip
import os
import struct
import tempfile

import numpy as np
import SimpleITK as sitk

GZIP_MAGIC = b"\x1f\x8b"
HEADER_BYTES = 348
SROW_OFFSET = 280


def is_gzipped(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def _raw_bytes(path: str) -> bytes:
    opener = gzip.open if is_gzipped(path) else open
    with opener(path, "rb") as f:
        return f.read()


def _read_bytes(data: bytes) -> sitk.Image:
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return sitk.ReadImage(tmp_path)
    finally:
        os.unlink(tmp_path)


def _orthonormalise_srow(data: bytes) -> bytes:
    srow = np.array(struct.unpack("<12f", data[SROW_OFFSET:SROW_OFFSET + 48])).reshape(3, 4)
    rot, offset = srow[:, :3], srow[:, 3]
    spacing = np.linalg.norm(rot, axis=0)
    u, _, vt = np.linalg.svd(rot / spacing)
    fixed = np.hstack([(u @ vt) * spacing, offset[:, None]])
    packed = struct.pack("<12f", *fixed.astype(np.float32).ravel())
    return data[:SROW_OFFSET] + packed + data[SROW_OFFSET + 48:]


def read_nifti(path: str) -> sitk.Image:
    if not is_gzipped(path) or path.endswith(".gz"):
        try:
            return sitk.ReadImage(path)
        except RuntimeError as e:
            if "orthonormal" not in str(e):
                raise
    data = _raw_bytes(path)
    try:
        return _read_bytes(data)
    except RuntimeError as e:
        if "orthonormal" not in str(e):
            raise
    # subject024's sform is rounded to 4 dp, which breaks ITK's orthonormality check
    return _read_bytes(_orthonormalise_srow(data))
