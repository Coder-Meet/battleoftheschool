"""CPU-only, parent-anchored detection of proximal arterial branches."""

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

FloatArray = npt.NDArray[np.float64]
Point = tuple[float, float, float]


@dataclass(frozen=True)
class DetectorConfig:
    spacing_mm: float = 1.0
    margin_mm: float = 20.0
    shell_inner_mm: float = 1.5
    shell_outer_mm: float = 3.5
    cap_margin_mm: float = 4.0
    minimum_radius_mm: float = 0.7
    minimum_path_mm: float = 5.0
    trace_length_mm: float = 10.0
    vesselness_floor: float = 0.06
    maximum_candidates: int = 160

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("All detector settings must be finite and positive.")
        if not self.shell_inner_mm < self.shell_outer_mm < self.margin_mm:
            raise ValueError("The candidate shell must fit inside the ROI margin.")
        if self.minimum_path_mm != 5 or self.trace_length_mm != 10:
            raise ValueError("The challenge requires a 5 mm seed and a 10 mm trace.")


@dataclass
class Branch:
    instance_id: str
    ostium_xyz_mm: Point
    seed_xyz_mm: Point
    radius_mm: float
    direction_xyz: Point
    path_xyz_mm: list[Point]
    evidence_score: float
    mean_vesselness: float
    parent_instance_id: str = "aorta"
    warnings: list[str] = field(default_factory=list)

    def prediction(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "parent_instance_id": self.parent_instance_id,
            "ostium_xyz_mm": self.ostium_xyz_mm,
            "seed_xyz_mm": self.seed_xyz_mm,
            "radius_mm": self.radius_mm,
            "direction_xyz": self.direction_xyz,
        }


@dataclass
class Detection:
    branches: list[Branch]
    timings: dict[str, float]
    blood_model: dict[str, float]
    candidates: int
    rejections: dict[str, int]
    warnings: list[str]
    config: DetectorConfig

    def prediction(self, case_id: str) -> dict:
        return {
            "case_id": case_id,
            "parent": {"instance_id": "aorta"},
            "daughters": [b.prediction() for b in self.branches],
        }

    def diagnostics(self) -> dict:
        return asdict(self)


def validate_geometry(image: sitk.Image, mask: sitk.Image) -> None:
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("CT and aorta mask must both be three-dimensional.")
    if image.GetNumberOfComponentsPerPixel() != 1 or mask.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("CT and mask must contain scalar voxels.")
    if image.GetSize() != mask.GetSize():
        raise ValueError("CT and aorta mask have different grid sizes.")
    for left, right in (
        (image.GetSpacing(), mask.GetSpacing()),
        (image.GetOrigin(), mask.GetOrigin()),
        (image.GetDirection(), mask.GetDirection()),
    ):
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("Image geometry contains non-finite values.")
        if not np.allclose(left, right, rtol=0, atol=1e-4):
            raise ValueError("CT and aorta mask do not share the same physical geometry.")
    if np.any(np.asarray(image.GetSpacing()) <= 0):
        raise ValueError("Image spacing must be positive.")
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-4):
        raise ValueError("Image direction must be orthonormal for physical distance measurements.")
    for volume in (image, mask):
        if not np.isfinite(sitk.GetArrayViewFromImage(volume)).all():
            raise ValueError("Input volume contains non-finite voxel values.")
    labels = np.unique(sitk.GetArrayViewFromImage(mask))
    if np.any(labels < 0) or len(labels[labels > 0]) > 1:
        raise ValueError("Supply a binary parent-aorta mask, not a multi-label segmentation.")


def physical_points(image: sitk.Image, points_zyx: npt.ArrayLike) -> FloatArray:
    points = np.asarray(points_zyx, dtype=float).reshape(-1, 3)[:, ::-1]
    origin = np.asarray(image.TransformIndexToPhysicalPoint((0, 0, 0)))
    basis = np.column_stack([
        np.asarray(image.TransformIndexToPhysicalPoint(tuple(int(j == i) for j in range(3))))
        - origin
        for i in range(3)
    ])
    return points @ basis.T + origin


def prepare_roi(
    image: sitk.Image, mask: sitk.Image, config: DetectorConfig
) -> tuple[sitk.Image, npt.NDArray[np.bool_]]:
    indices = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)
    padding = np.ceil(config.margin_mm / np.asarray(image.GetSpacing())[::-1]).astype(int)
    lower = np.maximum(indices.min(axis=0) - padding, 0)[::-1]
    upper = np.minimum(
        indices.max(axis=0) + padding + 1, np.asarray(image.GetSize())[::-1]
    )[::-1]
    size = (upper - lower).tolist()
    ct_crop = sitk.RegionOfInterest(image, size, lower.tolist())
    mask_crop = sitk.RegionOfInterest(mask > 0, size, lower.tolist())
    new_size = np.ceil(
        (np.asarray(size) - 1) * image.GetSpacing() / config.spacing_mm
    ).astype(int) + 1
    grid = sitk.Image(new_size.tolist(), sitk.sitkFloat32)
    grid.SetOrigin(ct_crop.GetOrigin())
    grid.SetDirection(image.GetDirection())
    grid.SetSpacing((config.spacing_mm,) * 3)
    ct = sitk.Resample(ct_crop, grid, sitk.Transform(), sitk.sitkLinear, -1024)
    parent = sitk.GetArrayFromImage(sitk.Resample(
        mask_crop, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )) > 0
    return ct, parent


def parent_curve(parent: npt.NDArray[np.bool_], spacing: float) -> FloatArray:
    indices = np.argwhere(parent)
    if not len(indices):
        return np.empty((0, 3), dtype=float)
    axis = int(np.argmax(np.ptp(indices, axis=0)))
    inner = ndi.distance_transform_edt(parent)
    points = []
    for level in np.unique(indices[:, axis]):
        selection: list[slice | int] = [slice(None)] * 3
        selection[axis] = int(level)
        plane = tuple(selection)
        coords = np.argwhere(parent[plane])
        weights = inner[plane][parent[plane]] ** 2
        center = np.average(coords, axis=0, weights=weights)
        # Keep the navigation curve inside the supplied lumen.
        nearest = np.argmin(np.sum((coords - center) ** 2, axis=1))
        points.append(np.insert(coords[nearest], axis, level))
    curve = np.asarray(points, dtype=float)
    smoothed = ndi.gaussian_filter1d(curve, max(1, 2 / spacing), axis=0)
    rounded = np.rint(smoothed).astype(int)
    valid = parent[tuple(rounded.T)]
    curve[valid] = smoothed[valid]
    return curve


def cap_mask(parent: npt.NDArray[np.bool_], config: DetectorConfig) -> npt.NDArray[np.bool_]:
    labels, count = ndi.label(parent)
    excluded = np.zeros(parent.shape, dtype=bool)
    for label_id, slices in enumerate(ndi.find_objects(labels), 1):
        if slices is None:
            continue
        size = np.asarray([s.stop - s.start for s in slices])
        axis = int(np.argmax(size))
        width = max(1, int(np.ceil(config.cap_margin_mm / config.spacing_mm)))
        # Each substantial parent component has its own acquisition endpoints.
        if np.count_nonzero(labels[slices] == label_id) < 8:
            continue
        pad = int(np.ceil(config.margin_mm / config.spacing_mm))
        for start, stop in (
            (slices[axis].start - pad, slices[axis].start + width),
            (slices[axis].stop - width, slices[axis].stop + pad),
        ):
            section = [slice(None)] * 3
            section[axis] = slice(max(0, start), min(parent.shape[axis], stop))
            other_axes = [i for i in range(3) if i != axis]
            for i in other_axes:
                section[i] = slice(max(0, slices[i].start - pad), slices[i].stop + pad)
            excluded[tuple(section)] = True
    return excluded


def sample_at(array: npt.NDArray, point: npt.ArrayLike) -> float:
    return float(ndi.map_coordinates(array, np.asarray(point).reshape(3, 1), order=1)[0])


def point_tuple(point: npt.ArrayLike) -> Point:
    x, y, z = np.asarray(point, dtype=float)
    return float(x), float(y), float(z)


def truncate_path(path: FloatArray, length: float) -> FloatArray:
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if arc[-1] <= length:
        return path
    index = int(np.searchsorted(arc, length))
    ratio = (length - arc[index - 1]) / (arc[index] - arc[index - 1])
    return np.vstack((path[:index], path[index - 1] + ratio * (path[index] - path[index - 1])))


def branch_junctions(support: npt.NDArray, parent: npt.NDArray, spacing: float) -> FloatArray:
    skeleton = skeletonize(support)
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8)) - skeleton
    labels, _ = ndi.label(skeleton & ~parent & (degree >= 3), structure=np.ones((3, 3, 3)))
    junctions = []
    padding = int(np.ceil(5 / spacing))
    for label_id, region in enumerate(ndi.find_objects(labels), 1):
        if region is None:
            continue
        nodes = np.argwhere(labels[region] == label_id) + np.asarray([s.start for s in region])
        center = nodes.mean(axis=0)
        low = np.maximum(np.floor(center).astype(int) - padding, 0)
        high = np.minimum(np.ceil(center).astype(int) + padding + 1, parent.shape)
        area = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        coordinates = np.moveaxis(np.indices(skeleton[area].shape), 0, -1) + low
        distances = np.linalg.norm(coordinates - center, axis=-1) * spacing
        arms, _ = ndi.label(
            skeleton[area] & (distances > 2) & (distances <= 5),
            structure=np.ones((3, 3, 3)),
        )
        substantial = 0
        for arm in np.unique(arms):
            if arm == 0:
                continue
            radii = distances[arms == arm]
            substantial += int(radii.min() <= 3.5 and radii.max() >= 4)
        if substantial >= 3:
            junctions.append(center)
    return np.asarray(junctions, dtype=float).reshape(-1, 3)


def stop_at_junction(path_mm: FloatArray, junctions_mm: FloatArray, tolerance_mm: float) -> FloatArray:
    segment = np.diff(path_mm, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    stop = cumulative[-1]
    for junction in junctions_mm:
        fraction = np.clip(
            np.sum((junction - path_mm[:-1]) * segment, axis=1) / np.maximum(lengths**2, 1e-8), 0, 1
        )
        projections = path_mm[:-1] + fraction[:, None] * segment
        distances = np.linalg.norm(projections - junction, axis=1)
        closest = int(np.argmin(distances))
        along = cumulative[closest] + fraction[closest] * lengths[closest]
        if distances[closest] <= tolerance_mm and along > 0:
            stop = min(stop, along)
    return truncate_path(path_mm, stop)


def _trace(
    root: npt.NDArray[np.int64],
    outside: npt.NDArray,
    parent: npt.NDArray,
    support: npt.NDArray,
    radius: npt.NDArray,
    vesselness: npt.NDArray,
    junctions: FloatArray,
    signed_distance: npt.NDArray,
    grid: sitk.Image,
    config: DetectorConfig,
) -> tuple[Branch | None, str]:
    spacing = config.spacing_mm
    pad = int(np.ceil(15 / spacing))
    low = np.maximum(root - pad, 0)
    high = np.minimum(root + pad + 1, parent.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    local_support = support[region]
    local_radius = radius[region]
    root_local = root - low
    cost = 1 / (0.5 + local_radius) + 0.7 * (1 - vesselness[region])
    cost[~local_support] = np.inf
    cost[parent[region]] = np.inf
    solver = MCP_Geometric(cost, sampling=(spacing,) * 3)
    cumulative, _ = solver.find_costs([tuple(root_local)])
    endpoints = (
        np.isfinite(cumulative)
        & (outside[region] >= config.minimum_path_mm + spacing / 2)
        & (outside[region] <= 12)
        & (local_radius >= config.minimum_radius_mm)
    )
    if not endpoints.any():
        return None, "no_supported_5mm_path"
    quality = cumulative / np.maximum(outside[region] - outside[tuple(root)], 1)
    quality[~endpoints] = np.inf
    endpoint = np.unravel_index(np.argmin(quality), quality.shape)
    path = np.asarray(solver.traceback(endpoint), dtype=float) + low
    if len(path) < 3:
        return None, "short_path"

    search_low = np.maximum(root - int(np.ceil(6 / spacing)), 0)
    search_high = np.minimum(root + int(np.ceil(6 / spacing)) + 1, parent.shape)
    search = tuple(slice(int(a), int(b)) for a, b in zip(search_low, search_high))
    wall = np.argwhere(parent[search]) + search_low
    if not len(wall):
        return None, "no_parent_contact"
    nearest = wall[np.argmin(np.sum((wall - root) ** 2, axis=1))]
    connector = np.linspace(nearest, root, max(3, int(np.linalg.norm(root - nearest) * 3)))
    if np.any(ndi.map_coordinates(support.astype(np.uint8), connector.T, order=0) == 0):
        return None, "disconnected_ostium"
    values = ndi.map_coordinates(signed_distance, connector.T, order=1)
    crossing = np.flatnonzero(values >= 0)
    if not len(crossing) or crossing[0] == 0:
        return None, "no_wall_crossing"
    i = crossing[0]
    fraction = -values[i - 1] / (values[i] - values[i - 1])
    ostium = connector[i - 1] + fraction * (connector[i] - connector[i - 1])
    path = np.vstack((ostium, path))
    path_mm = physical_points(grid, path)
    path_mm = truncate_path(path_mm, config.trace_length_mm)
    before_junction = np.linalg.norm(np.diff(path_mm, axis=0), axis=1).sum()
    path_mm = stop_at_junction(path_mm, physical_points(grid, junctions), 1.5 * spacing)
    length = np.linalg.norm(np.diff(path_mm, axis=0), axis=1).sum()
    if length < config.minimum_path_mm:
        return None, "short_proximal_segment"
    seed = truncate_path(path_mm, config.minimum_path_mm)[-1]
    ost = path_mm[0]
    direction = seed - ost
    displacement = np.linalg.norm(direction)
    if displacement < 3.5:
        return None, "wall_hugging_path"
    direction /= displacement
    seed_index = np.asarray(grid.TransformPhysicalPointToContinuousIndex(seed.tolist()))[::-1]
    seed_radius = sample_at(radius, seed_index)
    if seed_radius < config.minimum_radius_mm:
        return None, "small_radius"
    if seed_radius > 8:
        return None, "wide_nonarterial_region"
    sampled_path = np.asarray([
        grid.TransformPhysicalPointToContinuousIndex(p.tolist()) for p in path_mm[1:]
    ])[:, ::-1]
    mean_vesselness = float(np.mean(ndi.map_coordinates(vesselness, sampled_path.T, order=1)))
    if mean_vesselness < config.vesselness_floor:
        return None, "weak_tubularity"
    score = float(np.clip(
        0.4 * min(mean_vesselness / 0.5, 1) + 0.35 * min(displacement / 5, 1)
        + 0.25 * min(length / 10, 1), 0, 1
    ))
    return Branch(
        instance_id="",
        ostium_xyz_mm=point_tuple(ost),
        seed_xyz_mm=point_tuple(seed),
        radius_mm=round(seed_radius, 3),
        direction_xyz=point_tuple(direction),
        path_xyz_mm=[point_tuple(p) for p in path_mm],
        evidence_score=round(score, 3),
        mean_vesselness=round(mean_vesselness, 3),
        warnings=["Trace stops at an estimated downstream bifurcation."] if length < before_junction - 0.01 else [],
    ), ""


def detect(
    image: sitk.Image, mask: sitk.Image, config: DetectorConfig | None = None
) -> Detection:
    config = config or DetectorConfig()
    start = perf_counter()
    validate_geometry(image, mask)
    warnings = [
        "Experimental detector: evidence scores are not calibrated probabilities.",
        "No daughter reference labels supplied; clinical accuracy is unmeasured.",
    ]
    if not np.any(sitk.GetArrayViewFromImage(mask) > 0):
        return Detection([], {"total_s": perf_counter() - start}, {}, 0, {}, warnings, config)
    grid, parent = prepare_roi(image, mask, config)
    if not parent.any():
        raise ValueError("Aorta mask vanished on the working grid; use finer spacing.")
    ct = sitk.GetArrayFromImage(grid)
    spacing = config.spacing_mm
    outside = ndi.distance_transform_edt(~parent, sampling=spacing)
    inside = ndi.distance_transform_edt(parent, sampling=spacing)
    smooth = ndi.gaussian_filter(ct, sigma=0.6 / spacing)
    core = smooth[inside >= 2]
    if len(core) < 20:
        core = smooth[parent]
    median = float(np.median(core))
    mad = float(np.median(np.abs(core - median)) * 1.4826)
    lower = max(30.0, median - max(65.0, 2.5 * mad))
    background = smooth[(outside >= 8) & (outside <= 16)]
    background_median = float(np.median(background)) if len(background) else median
    if background_median < median:
        lower = max(lower, (median + background_median) / 2)
    upper = median + max(120.0, 3.5 * mad)
    if median < 120:
        warnings.append("Low parent contrast: soft tissue and veins may mimic daughter arteries.")
    blood = {
        "median_hu": median, "mad_hu": mad, "lower_hu": lower, "upper_hu": upper,
        "background_median_hu": background_median,
    }
    prepared = perf_counter()
    normalized = np.clip((smooth - lower) / max(upper - lower, 1), 0, 1)
    tubular = sato(normalized, sigmas=(0.8 / spacing, 1.5 / spacing, 2.5 / spacing), black_ridges=False)
    shell_region = (outside > 0) & (outside <= 15)
    peak = float(np.percentile(tubular[shell_region], 99.5))
    tubular = np.clip(tubular / max(peak, 1e-6), 0, 1).astype(np.float32)
    support = (
        (smooth >= lower) & (smooth <= upper)
        & ((tubular >= config.vesselness_floor) | (outside <= 1.5))
        & (outside <= 16)
    ) | parent
    radius = ndi.distance_transform_edt(support, sampling=spacing).astype(np.float32)
    junctions = branch_junctions(support, parent, spacing)
    signed_distance = outside - inside
    excluded = cap_mask(parent, config)
    shell = (
        support & ~excluded
        & (outside >= config.shell_inner_mm) & (outside <= config.shell_outer_mm)
        & (tubular >= config.vesselness_floor)
    )
    labels, count = ndi.label(shell, structure=np.ones((3, 3, 3), dtype=int))
    enhanced = perf_counter()
    branches: list[Branch] = []
    rejections: dict[str, int] = {}
    candidates = []
    for label_id, region in enumerate(ndi.find_objects(labels), 1):
        if region is None:
            continue
        points = np.argwhere(labels[region] == label_id)
        points += np.asarray([s.start for s in region])
        if len(points) * spacing**3 < 3:
            continue
        quality = radius[tuple(points.T)] * (0.3 + tubular[tuple(points.T)])
        root = points[int(np.argmax(quality))]
        candidates.append((float(np.max(quality)), root, len(points) * spacing**3))
    candidates.sort(key=lambda c: (-c[0], tuple(c[1])))
    if len(candidates) > config.maximum_candidates:
        warnings.append(f"Candidate limit reached ({config.maximum_candidates}); weaker contacts were omitted.")
    for _, root, volume in candidates[:config.maximum_candidates]:
        if volume > 1200:
            reason = "broad_wall_contact"
            branch = None
        else:
            branch, reason = _trace(
                root, outside, parent, support, radius, tubular, junctions, signed_distance, grid, config
            )
        if branch is None:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue
        duplicate = any(
            np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - old.ostium_xyz_mm) < 2.5
            and np.linalg.norm(np.asarray(branch.seed_xyz_mm) - old.seed_xyz_mm) < 3
            for old in branches
        )
        if duplicate:
            rejections["same_opening_and_path"] = rejections.get("same_opening_and_path", 0) + 1
        else:
            branches.append(branch)
    branches.sort(key=lambda b: (b.ostium_xyz_mm[2], b.ostium_xyz_mm[1], b.ostium_xyz_mm[0]))
    for number, branch in enumerate(branches, 1):
        branch.instance_id = f"branch_{number:03d}"
    return Detection(
        branches,
        {
            "prepare_s": round(prepared - start, 3),
            "enhance_s": round(enhanced - prepared, 3),
            "trace_s": round(perf_counter() - enhanced, 3),
            "total_s": round(perf_counter() - start, 3),
        },
        blood, len(candidates), rejections, warnings, config,
    )
