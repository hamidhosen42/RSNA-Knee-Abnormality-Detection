"""Synthetic DICOM knee phantom + acquisitions in patient LPS coordinates (for the mirror TTA tests)."""
import os

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

M = np.diag([-1.0, 1.0, 1.0])   # patient left-right mirror (LPS x -> -x)

# asymmetric blobs (x, y, z, sigma, amplitude): no symmetry in any axis
BLOBS = [(25, -10, 5, 9, 900), (-30, 15, -12, 12, 500), (5, 30, 25, 7, 700), (-8, -35, -30, 10, 300),
         (40, 20, -5, 6, 650), (-45, -5, 35, 8, 450), (12, 5, -40, 11, 380)]


def phantom(p, mirrored=False):
    """p [..., 3] mm in LPS -> intensity; mirrored=True gives the patient's mirror image f(Mp)."""
    p = p @ M.T if mirrored else p
    v = 60.0 + 0.8 * p[..., 0] + 0.5 * p[..., 1] - 0.3 * p[..., 2] + 120.0
    for x, y, z, s, a in BLOBS:
        d2 = (p[..., 0] - x) ** 2 + (p[..., 1] - y) ** 2 + (p[..., 2] - z) ** 2
        v = v + a * np.exp(-d2 / (2 * s * s))
    return np.clip(v, 0, 4000)


def rot(axis, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ds_round(v):
    """DICOM DS-like rounding (what the file stores)."""
    return [float(f"{x:.6f}") for x in v]


def acquisition(r, c, rows, cols, ps_row, ps_col, n_slices, gap, centre=(0.0, 0.0, 0.0), slice_sign=1.0):
    """Returns dict with rounded IOP, per-slice IPP list, pixel-centre grid P [t, i, j, 3] (from the rounded geometry)."""
    r, c = np.array(ds_round(r)), np.array(ds_round(c))
    n = np.cross(r, c)
    centre = np.array(centre, float)
    offs = (np.arange(n_slices) - (n_slices - 1) / 2) * gap * slice_sign
    ipps = [ds_round(centre + o * n - (cols - 1) / 2 * ps_col * r - (rows - 1) / 2 * ps_row * c) for o in offs]
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    P = np.stack([np.array(ipp)[None, None, :] + jj[..., None] * ps_col * r + ii[..., None] * ps_row * c for ipp in ipps])
    return {"iop": list(r) + list(c), "ipps": ipps, "P": P, "ps": [ps_row, ps_col]}


def write_series(d, arrays, iop, ipps, ps, inst_order=None, drop_iop=False, drop_ipp=False):
    os.makedirs(d, exist_ok=True)
    n = len(arrays)
    inst = list(range(1, n + 1)) if inst_order is None else inst_order
    for t in range(n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = MRImageStorage, meta.MediaStorageSOPInstanceUID, "MR"
        a = np.ascontiguousarray(arrays[t]).astype(np.uint16)
        ds.Rows, ds.Columns = a.shape
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        if not drop_iop:
            ds.ImageOrientationPatient = [f"{v:.6f}" for v in iop]
        if not drop_ipp:
            ds.ImagePositionPatient = [f"{v:.6f}" for v in ipps[t]]
        ds.PixelSpacing = [f"{ps[0]:.6f}", f"{ps[1]:.6f}"]
        ds.InstanceNumber = int(inst[t])
        ds.PixelData = a.tobytes()
        ds.save_as(os.path.join(d, f"im{t:03d}.dcm"), enforce_file_format=True)


def mirrored_variant(acq, arrays, variant):
    """The same physical points, patient-mirrored, written in another scanner convention.
    v0: r'=Mr, c'=Mc (same arrays); v1: r'=-Mr (columns reversed); v2: transposed (r'=Mc, c'=Mr);
    v3: v0 with the instance numbers running the other way."""
    r, c = np.array(acq["iop"][:3]), np.array(acq["iop"][3:])
    P = acq["P"]
    if variant in ("v0", "v3"):
        iop = list(M @ r) + list(M @ c)
        ipps = [ds_round(M @ P[t, 0, 0]) for t in range(len(P))]
        return arrays, iop, ipps, acq["ps"]
    if variant == "v1":
        iop = list(-(M @ r)) + list(M @ c)
        ipps = [ds_round(M @ P[t, 0, -1]) for t in range(len(P))]
        return [a[:, ::-1] for a in arrays], iop, ipps, acq["ps"]
    if variant == "v2":
        iop = list(M @ c) + list(M @ r)
        ipps = [ds_round(M @ P[t, 0, 0]) for t in range(len(P))]
        return [a.T for a in arrays], iop, ipps, acq["ps"][::-1]
    raise ValueError(variant)
