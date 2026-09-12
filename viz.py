#!/usr/bin/env python3
"""
Visual check for a Branchseed case.

Renders orthogonal slices through the aorta plus coronal/sagittal MIPs of the
region around the mask, with the aorta outline and (optionally) predicted
ostia + direction arrows from a prediction JSON overlaid.

    python viz.py --subject subject001
    python viz.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --output viz/subject001.png
    python viz.py --subject subject001 --pred prediction.json
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CT_WINDOW = (-100, 600)
MIP_WINDOW = (100, 600)
MARGIN_MM = 25.0
ARROW_MM = 10.0


def parse_args():
    p = argparse.ArgumentParser(description="Render a visual check for one case.")
    p.add_argument("--subject", default=None, help="e.g. subject001; resolves image/mask/output under data/ and viz/")
    p.add_argument("--image", default=None)
    p.add_argument("--aorta-mask", default=None)
    p.add_argument("--output", default=None, help="PNG path to write (default viz/<subject>.png)")
    p.add_argument("--pred", default=None, help="Optional prediction JSON to overlay")
    args = p.parse_args()
    if args.subject:
        args.subject = args.subject.strip("-")
        repo = os.path.dirname(os.path.abspath(__file__))
        args.image = args.image or find_one(os.path.join(repo, "data", args.subject, "orig*.nii*"))
        args.aorta_mask = args.aorta_mask or find_one(os.path.join(repo, "data", args.subject, "mask*.nii*"))
        args.output = args.output or os.path.join(repo, "viz", f"{args.subject}.png")
    if not (args.image and args.aorta_mask and args.output):
        p.error("pass --subject, or all of --image, --aorta-mask and --output")
    return args


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one file for {pattern}, found {hits}")
    return hits[0]


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def ensure_lfs_pulled(paths):
    pointers = [p for p in paths if is_lfs_pointer(p)]
    if not pointers:
        return
    repo = os.path.dirname(os.path.abspath(__file__))
    rel = [os.path.relpath(p, repo) for p in pointers]
    print(f"[lfs] pulling {len(rel)} file(s): {rel}")
    subprocess.run(["git", "lfs", "pull", "--include=" + ",".join(rel)], cwd=repo, check=True)
    still = [p for p in pointers if is_lfs_pointer(p)]
    if still:
        raise SystemExit(f"still LFS pointers after pull (is git-lfs installed?): {still}")


def describe(name, img, arr):
    print(f"[{name}] size xyz={img.GetSize()} spacing={tuple(round(s, 3) for s in img.GetSpacing())} "
          f"origin={tuple(round(o, 1) for o in img.GetOrigin())} dtype={arr.dtype} "
          f"range=({arr.min()}, {arr.max()})")


def mask_bbox_zyx(mask_zyx):
    zs, ys, xs = np.nonzero(mask_zyx)
    return (zs.min(), zs.max() + 1), (ys.min(), ys.max() + 1), (xs.min(), xs.max() + 1)


def pad_bbox(bbox, shape, spacing_zyx):
    out = []
    for (lo, hi), n, sp in zip(bbox, shape, spacing_zyx):
        pad = int(round(MARGIN_MM / sp))
        out.append((max(0, lo - pad), min(n, hi + pad)))
    return out


def window(arr, lo, hi):
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def load_pred_points(path, image):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        pred = json.load(f)
    pts = []
    for d in pred.get("daughters", []):
        ost = np.array(image.TransformPhysicalPointToContinuousIndex(d["ostium_xyz_mm"]))
        seed = np.array(image.TransformPhysicalPointToContinuousIndex(d["seed_xyz_mm"]))
        tip_mm = np.array(d["ostium_xyz_mm"]) + ARROW_MM * np.array(d["direction_xyz"])
        tip = np.array(image.TransformPhysicalPointToContinuousIndex(tip_mm.tolist()))
        pts.append({"id": d["instance_id"], "ost": ost, "seed": seed, "tip": tip, "r": d.get("radius_mm")})
    return pts


def draw_slice(ax, ct2d, mask2d, extent, title):
    ax.imshow(ct2d, cmap="gray", origin="lower", extent=extent, aspect="equal")
    if mask2d.any():
        ax.contour(mask2d, levels=[0.5], colors="cyan", linewidths=0.8, origin="lower", extent=extent)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_points(ax, pts, axes_idx, offset, spacing):
    # axes_idx = (horizontal index axis, vertical index axis) in xyz order
    h, v = axes_idx
    for p in pts:
        ox = (p["ost"][h] - offset[h]) * spacing[h]
        oy = (p["ost"][v] - offset[v]) * spacing[v]
        tx = (p["tip"][h] - offset[h]) * spacing[h]
        ty = (p["tip"][v] - offset[v]) * spacing[v]
        ax.plot(ox, oy, "o", color="yellow", ms=5, mec="black")
        ax.annotate("", xy=(tx, ty), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax.text(ox, oy, f" {p['id'][-3:]}", color="yellow", fontsize=7)


def main():
    args = parse_args()
    ensure_lfs_pulled([args.image, args.aorta_mask])
    image = sitk.ReadImage(args.image)
    mask = sitk.ReadImage(args.aorta_mask)

    ct = sitk.GetArrayFromImage(image).astype(np.float32)  # zyx
    mk = sitk.GetArrayFromImage(mask) > 0
    describe("image", image, ct)
    describe("mask", mask, mk.astype(np.uint8))
    sx, sy, sz = image.GetSpacing()
    spacing_zyx = (sz, sy, sx)

    if not mk.any():
        raise SystemExit("mask is empty")

    bbox = mask_bbox_zyx(mk)
    (z0, z1), (y0, y1), (x0, x1) = pad_bbox(bbox, ct.shape, spacing_zyx)
    print(f"[mask] voxels={mk.sum()} bbox zyx={bbox} -> crop z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]")
    print(f"[mask] mean HU inside aorta={ct[mk].mean():.1f} median={np.median(ct[mk]):.1f}")

    ctc = ct[z0:z1, y0:y1, x0:x1]
    mkc = mk[z0:z1, y0:y1, x0:x1]
    offset_xyz = (x0, y0, z0)
    spacing_xyz = (sx, sy, sz)

    cz, cy, cx = [int(round(v)) for v in np.array(np.nonzero(mkc)).mean(axis=1)]
    ext_xy = (0, ctc.shape[2] * sx, 0, ctc.shape[1] * sy)
    ext_xz = (0, ctc.shape[2] * sx, 0, ctc.shape[0] * sz)
    ext_yz = (0, ctc.shape[1] * sy, 0, ctc.shape[0] * sz)

    pts = load_pred_points(args.pred, image)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    w = window(ctc, *CT_WINDOW)
    draw_slice(axes[0, 0], w[cz], mkc[cz], ext_xy, f"axial z={cz + z0}")
    draw_slice(axes[0, 1], w[:, cy, :], mkc[:, cy, :], ext_xz, f"coronal y={cy + y0}")
    draw_slice(axes[0, 2], w[:, :, cx], mkc[:, :, cx], ext_yz, f"sagittal x={cx + x0}")

    m = window(ctc, *MIP_WINDOW)
    draw_slice(axes[1, 1], m.max(axis=1), mkc.max(axis=1), ext_xz, "coronal MIP (contrast window)")
    draw_slice(axes[1, 2], m.max(axis=2), mkc.max(axis=2), ext_yz, "sagittal MIP (contrast window)")
    draw_slice(axes[1, 0], m.max(axis=0), mkc.max(axis=0), ext_xy, "axial MIP (contrast window)")

    if pts:
        draw_points(axes[1, 0], pts, (0, 1), offset_xyz, spacing_xyz)
        draw_points(axes[1, 1], pts, (0, 2), offset_xyz, spacing_xyz)
        draw_points(axes[1, 2], pts, (1, 2), offset_xyz, spacing_xyz)

    case = os.path.basename(os.path.dirname(os.path.abspath(args.image)))
    fig.suptitle(f"{case}  |  cyan = aorta mask, yellow = ostium, red = direction  |  {len(pts)} daughters", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
