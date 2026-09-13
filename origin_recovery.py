"""Experimental recovery of supported, curved connectors to the parent wall."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

from detector import (
    Branch, Candidate, Detection, DetectorConfig, NormalizedScan, VesselEvidence,
    _trace, detect, enhance, normalize, propose, resolve, validate_geometry,
)


def follows_root(
    branch: Branch, root: npt.NDArray[np.int64], grid: sitk.Image, root_radius_mm: float,
) -> bool:
    point = np.asarray(grid.TransformContinuousIndexToPhysicalPoint(root[::-1].astype(float).tolist()))
    path = np.asarray(branch.path_xyz_mm)
    segments = np.diff(path, axis=0)
    fraction = np.clip(
        np.sum((point - path[:-1]) * segments, axis=1)
        / np.maximum(np.sum(segments**2, axis=1), 1e-8), 0, 1,
    )
    distance = np.linalg.norm(path[:-1] + fraction[:, None] * segments - point, axis=1)
    return bool(distance.min() <= root_radius_mm)


def connected_wall_root(
    root: npt.NDArray[np.int64], parent: npt.NDArray[np.bool_],
    support: npt.NDArray[np.bool_], excluded: npt.NDArray[np.bool_], spacing_mm: float,
) -> npt.NDArray[np.int64] | None:
    pad = int(np.ceil(6.0 / spacing_mm))
    low = np.maximum(root - pad, 0)
    high = np.minimum(root + pad + 1, parent.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    lumen = support[region] & ~parent[region] & ~excluded[region]
    contact = lumen & ndi.binary_dilation(
        parent[region], structure=ndi.generate_binary_structure(3, 1),
    )
    if not contact.any() or not lumen[tuple(root - low)]:
        return None
    cost = np.where(lumen, 1.0, np.inf)
    solver = MCP_Geometric(cost, sampling=(spacing_mm,) * 3, fully_connected=False)
    distance, _ = solver.find_costs([tuple(root - low)])
    distance[~contact | (distance > 6.0)] = np.inf
    if not np.isfinite(distance).any():
        return None
    endpoint = np.asarray(np.unravel_index(np.argmin(distance), distance.shape))
    return (endpoint + low).astype(np.int64)


def recovery_candidates(
    scan: NormalizedScan, evidence: VesselEvidence, candidates: list[Candidate],
    config: DetectorConfig, require_root: bool = False,
) -> list[Candidate]:
    additions: list[Candidate] = []
    for quality, root, volume in candidates[:config.maximum_candidates]:
        if volume > config.broad_contact_mm3:
            continue
        _, reason = _trace(
            root, scan.outside, scan.parent, evidence.support, evidence.radius, evidence.tubular,
            evidence.junctions, evidence.signed_distance, evidence.context, volume, scan.grid, config,
        )
        if reason != "disconnected_ostium":
            continue
        relocated = connected_wall_root(
            root, scan.parent, evidence.support, evidence.excluded, config.spacing_mm,
        )
        if relocated is not None and require_root:
            branch, _ = _trace(
                relocated, scan.outside, scan.parent, evidence.support, evidence.radius, evidence.tubular,
                evidence.junctions, evidence.signed_distance, evidence.context, volume, scan.grid, config,
            )
            if branch is None or not follows_root(
                branch, root, scan.grid, float(evidence.radius[tuple(root)]),
            ):
                continue
        if relocated is not None and not any(
            np.array_equal(relocated, other) for _, other, _ in candidates + additions
        ):
            additions.append((quality, relocated, volume))
    return additions


def detect_connected(
    image: sitk.Image, mask: sitk.Image, config: DetectorConfig | None = None, *,
    require_root: bool = False,
) -> Detection:
    profile = "experimental-guarded-origin" if require_root else "experimental-connected-origin"
    config = replace(config or DetectorConfig(), profile=profile)
    validate_geometry(image, mask)
    if not np.any(sitk.GetArrayViewFromImage(mask) > 0):
        return detect(image, mask, config)
    start = perf_counter()
    scan = normalize(image, mask, config)
    prepared = perf_counter()
    evidence = enhance(scan, config)
    candidates, warnings = propose(scan, evidence, config)
    candidates = candidates[:config.maximum_candidates]
    additions = recovery_candidates(scan, evidence, candidates, config, require_root)
    additions = additions[:max(0, config.maximum_candidates - len(candidates))]
    enhanced = perf_counter()
    branches, rejections = resolve(scan, evidence, candidates + additions, config)
    return Detection(
        branches,
        {"prepare_s": prepared - start, "enhance_s": enhanced - prepared,
         "trace_s": perf_counter() - enhanced, "total_s": perf_counter() - start},
        scan.blood, len(candidates) + len(additions),
        {**rejections, "connected_wall_roots": len(additions)},
        scan.warnings + warnings + ["Experimental recovery; not the default submission algorithm."],
        config,
    )
