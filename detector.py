"""CPU-only, parent-anchored detection of proximal arterial branches.

Pipeline stages, in order: normalize (grid, smoothing, blood window) -> enhance (tubularity, support, radius)
-> propose (wall-contact roots) -> resolve (trace, filter, measure). See detect() for the orchestration."""

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
    shell_vesselness_floor: float = 0.06
    connector_gap_fraction: float = 0.0
    root_depth_mm: float = 3.5
    blood_lower_scale: float = 1.0
    support_contrast_fraction: float = 0.5
    native_contrast_scale: float = 1.2
    roots_per_contact: int = 1
    wall_hug_penalty: float = 0.0
    broad_contact_mm3: float = 1200.0
    parallel_clearance_mm: float = 0.0
    profile: str = "strict"

    def __post_init__(self) -> None:
        numeric = {k: v for k, v in asdict(self).items() if k != "profile"}
        values = np.asarray(list(numeric.values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("All detector settings must be finite.")
        may_be_zero = {"connector_gap_fraction", "wall_hug_penalty", "native_contrast_scale", "parallel_clearance_mm"}
        positive = [v for k, v in numeric.items() if k not in may_be_zero]
        if (
            np.any(np.asarray(positive, dtype=float) <= 0)
            or self.wall_hug_penalty < 0 or self.native_contrast_scale < 0 or self.parallel_clearance_mm < 0
        ):
            raise ValueError("All detector settings must be finite and positive.")
        if not 0 <= self.connector_gap_fraction < 1:
            raise ValueError("The connector gap fraction must lie in [0, 1).")
        if not 0 < self.support_contrast_fraction <= 1:
            raise ValueError("The support contrast fraction must lie in (0, 1].")
        if not self.shell_inner_mm < self.shell_outer_mm < self.margin_mm:
            raise ValueError("The candidate shell must fit inside the ROI margin.")
        if self.minimum_path_mm != 5 or self.trace_length_mm != 10:
            raise ValueError("The challenge requires a 5 mm seed and a 10 mm trace.")

    @classmethod
    def review(cls, **overrides: float | str) -> "DetectorConfig":
        # Wide proposal pool for human labelling; the classifier can only remove, never add, candidates.
        settings: dict[str, float | str] = {
            "shell_outer_mm": 6.0, "vesselness_floor": 0.03, "shell_vesselness_floor": 0.005,
            "connector_gap_fraction": 0.35, "root_depth_mm": 2.5, "blood_lower_scale": 2.0,
            "roots_per_contact": 6, "wall_hug_penalty": 0.8, "broad_contact_mm3": 1e6,
            "profile": "review",
        }
        settings.update(overrides)
        return cls(**settings)  # type: ignore[arg-type]


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
    features: dict[str, float] = field(default_factory=dict)

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


def shares_prefix(path_mm: FloatArray, other_mm: FloatArray, prefix_mm: float = 3.0, tolerance_mm: float = 1.5) -> bool:
    """True when the first prefix_mm of path_mm runs within tolerance_mm of the other path: a shared trunk."""
    prefix = truncate_path(path_mm, prefix_mm)
    segments = np.diff(other_mm, axis=0)
    lengths = np.maximum(np.sum(segments**2, axis=1), 1e-8)
    for point in prefix:
        fraction = np.clip(np.sum((point - other_mm[:-1]) * segments, axis=1) / lengths, 0, 1)
        nearest = other_mm[:-1] + fraction[:, None] * segments
        if np.min(np.linalg.norm(nearest - point, axis=1)) > tolerance_mm:
            return False
    return True


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


def upstream_contact(
    path: FloatArray, geodesic: npt.NDArray, outside: npt.NDArray, config: DetectorConfig,
) -> FloatArray | None:
    """Where a wall-hugging daughter first enters the contact shell, upstream of the root along its own course."""
    spacing = config.spacing_mm
    proximal = truncate_path(path * spacing, 5) / spacing
    centered = proximal - proximal.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    tangent = vectors[0]
    if np.dot(tangent, proximal[-1] - proximal[0]) < 0:
        tangent = -tangent
    shell = (
        np.isfinite(geodesic)
        & (geodesic <= 6.0)
        & (outside >= config.shell_inner_mm)
        & (outside <= config.shell_outer_mm + spacing)
    )
    if not shell.any():
        return None
    voxels = np.argwhere(shell)
    along = (voxels - path[0]) @ tangent * spacing
    best = int(np.argmin(along))
    return voxels[best].astype(float) if along[best] < -spacing else None


def wall_origin(
    path: FloatArray, parent: npt.NDArray, support: npt.NDArray,
    signed_distance: npt.NDArray, spacing: float, gap_fraction: float = 0.0,
) -> FloatArray | None:
    proximal = truncate_path(path * spacing, 5) / spacing
    centered = proximal - proximal.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    tangent = vectors[0]
    if np.dot(tangent, proximal[-1] - proximal[0]) < 0:
        tangent = -tangent
    root = path[0]
    low = np.maximum(np.floor(root - 6 / spacing).astype(int), 0)
    high = np.minimum(np.ceil(root + 6 / spacing).astype(int) + 1, parent.shape)
    search = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    wall = np.argwhere(parent[search]) + low
    if not len(wall):
        return None
    nearest = wall[np.argmin(np.sum((wall - root) ** 2, axis=1))]
    for target in (root - tangent * 10 / spacing, nearest):
        steps = max(3, int(np.ceil(np.linalg.norm(target - root) * spacing / 0.25)) + 1)
        connector = np.linspace(root, target, steps)
        values = ndi.map_coordinates(signed_distance, connector.T, order=1, mode="constant", cval=np.inf)
        inside = np.flatnonzero(values <= 0)
        if not len(inside) or inside[0] == 0:
            continue
        index = inside[0]
        contact = connector[:index + 1]
        unsupported = ndi.map_coordinates(support.astype(np.uint8), contact.T, order=0) == 0
        if float(np.mean(unsupported)) > gap_fraction:
            continue
        fraction = values[index - 1] / (values[index - 1] - values[index])
        return connector[index - 1] + fraction * (connector[index] - connector[index - 1])
    return None


def cross_section_radius(
    intensity: npt.NDArray, seed: FloatArray, direction: FloatArray, spacing: float, level: float,
) -> float | None:
    axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, axis)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rays = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    distances = np.arange(0, 8.25, 0.25)
    points = seed[:, None, None] + np.moveaxis(rays, -1, 0)[:, :, None] * distances / spacing
    values = ndi.map_coordinates(intensity, points, order=1, mode="constant", cval=np.nan)
    if not np.all(values[:, 0] > level):
        return None
    radii = np.full(32, np.nan)
    for index, samples in enumerate(values):
        crossing = np.flatnonzero(samples <= level)
        if len(crossing):
            stop = crossing[0]
            fraction = (samples[stop - 1] - level) / (samples[stop - 1] - samples[stop])
            radii[index] = distances[stop - 1] + 0.25 * fraction
    diameters = radii[:16] + radii[16:]
    valid = diameters[np.isfinite(diameters)]
    return float(np.median(valid) / 2) if len(valid) >= 12 else None


@dataclass(frozen=True)
class TraceContext:
    intensity: npt.NDArray
    lumen_level: float
    bone_distance: npt.NDArray
    axis_zyx: FloatArray
    axis_range: tuple[float, float]
    native_spacing_mm: float
    median: float
    background_median: float


def connector_gap(support: npt.NDArray, start: FloatArray, end: FloatArray) -> float:
    steps = max(3, int(np.ceil(np.linalg.norm(end - start) * 4)) + 1)
    line = np.linspace(start, end, steps)
    return float(np.mean(ndi.map_coordinates(support.astype(np.uint8), line.T, order=0) == 0))


def parent_angle(parent: npt.NDArray, point: FloatArray, direction_zyx: FloatArray, spacing: float) -> float:
    pad = int(np.ceil(12 / spacing))
    low = np.maximum(np.floor(point).astype(int) - pad, 0)
    high = np.minimum(np.ceil(point).astype(int) + pad + 1, parent.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    voxels = np.argwhere(parent[region]).astype(float)
    if len(voxels) < 3:
        return 90.0
    _, _, vectors = np.linalg.svd(voxels - voxels.mean(axis=0), full_matrices=False)
    cosine = abs(float(np.dot(vectors[0], direction_zyx)))
    return float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))


def _trace(
    root: npt.NDArray[np.int64],
    outside: npt.NDArray,
    parent: npt.NDArray,
    support: npt.NDArray,
    radius: npt.NDArray,
    vesselness: npt.NDArray,
    junctions: FloatArray,
    signed_distance: npt.NDArray,
    ctx: TraceContext,
    volume_mm3: float,
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
    if config.wall_hug_penalty > 0:
        cost = cost + config.wall_hug_penalty * np.exp(-outside[region] / 2.0)
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
    quality = cumulative / np.maximum(outside[region] - outside[tuple(root)], 1)
    geodesic: npt.NDArray | None = None
    if not endpoints.any() and config.parallel_clearance_mm > 0:
        # A daughter that runs along the parent wall never gets 5 mm away from it; measure path length instead.
        unit = np.where(np.isfinite(cost), 1.0, np.inf)
        geodesic, _ = MCP_Geometric(unit, sampling=(spacing,) * 3).find_costs([tuple(root_local)])
        endpoints = (
            np.isfinite(geodesic)
            & (geodesic >= config.minimum_path_mm + spacing / 2)
            & (geodesic <= 12)
            & (outside[region] >= config.parallel_clearance_mm)
            & (local_radius >= config.minimum_radius_mm)
        )
        finite = np.isfinite(geodesic)
        quality = np.full_like(cumulative, np.inf)
        quality[finite] = cumulative[finite] / np.maximum(geodesic[finite], 1)
    if not endpoints.any():
        return None, "no_supported_5mm_path"
    quality[~endpoints] = np.inf
    endpoint = np.unravel_index(np.argmin(quality), quality.shape)
    path = np.asarray(solver.traceback(endpoint), dtype=float) + low
    if len(path) < 3:
        return None, "short_path"
    if geodesic is not None:
        upstream = upstream_contact(path - low, geodesic, outside[region], config)
        if upstream is not None:
            path = np.vstack((upstream + low, path))

    ostium = wall_origin(path, parent, support, signed_distance, spacing, config.connector_gap_fraction)
    if ostium is None:
        return None, "disconnected_ostium"
    gap = connector_gap(support, ostium, path[0])
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
    local_direction = (np.asarray(grid.GetDirection()).reshape(3, 3).T @ direction)[::-1]
    measured_radius = cross_section_radius(ctx.intensity, seed_index, local_direction, spacing, ctx.lumen_level)
    if measured_radius is not None:
        seed_radius = measured_radius
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
    hu_along = float(np.mean(ndi.map_coordinates(ctx.intensity, sampled_path.T, order=1)))
    if geodesic is not None and hu_along > ctx.median + 0.1 * max(ctx.median - ctx.background_median, 1.0):
        return None, "hyperdense_wall_structure"
    bone = float(np.min(ndi.map_coordinates(ctx.bone_distance, sampled_path.T, order=1)))
    span = max(ctx.axis_range[1] - ctx.axis_range[0], 1e-6)
    features = {
        "path_hu_relative": round((hu_along - ctx.background_median) / max(ctx.median - ctx.background_median, 1.0), 4),
        "bone_distance_mm": round(min(bone, 50.0), 3),
        "parent_angle_degrees": round(parent_angle(parent, ostium, local_direction, spacing), 2),
        "arc_position": round(float(np.clip((np.dot(ostium, ctx.axis_zyx) - ctx.axis_range[0]) / span, 0, 1)), 4),
        "native_spacing_mm": round(ctx.native_spacing_mm, 3),
        "connector_gap": round(gap, 4),
        "candidate_volume_mm3": round(volume_mm3, 2),
    }
    score = float(np.clip(
        (0.4 * min(mean_vesselness / 0.5, 1) + 0.35 * min(displacement / 5, 1)
         + 0.25 * min(length / 10, 1)) * (1 - gap), 0, 1
    ))
    warnings = ["Trace stops at an estimated downstream bifurcation."] if length < before_junction - 0.01 else []
    if gap > 0:
        warnings.append("Wall connector crosses unsupported voxels; verify the ostium on CT.")
    return Branch(
        instance_id="",
        ostium_xyz_mm=point_tuple(ost),
        seed_xyz_mm=point_tuple(seed),
        radius_mm=round(seed_radius, 3),
        direction_xyz=point_tuple(direction),
        path_xyz_mm=[point_tuple(p) for p in path_mm],
        evidence_score=round(score, 3),
        mean_vesselness=round(mean_vesselness, 3),
        warnings=warnings,
        features=features,
    ), ""


def detect_pool(image: sitk.Image, mask: sitk.Image, review: DetectorConfig | None = None) -> Detection:
    """Loose-profile candidates plus any strict detections they missed, so review never drops a submission branch."""
    strict = detect(image, mask)
    pool = detect(image, mask, review or DetectorConfig.review())
    added = 0
    for branch in strict.branches:
        if all(np.linalg.norm(np.subtract(branch.ostium_xyz_mm, b.ostium_xyz_mm)) >= 3 for b in pool.branches):
            branch.warnings = [*branch.warnings, "Strict-profile detection that the loose pool did not propose."]
            pool.branches.append(branch)
            added += 1
    pool.branches.sort(key=lambda b: (b.ostium_xyz_mm[2], b.ostium_xyz_mm[1], b.ostium_xyz_mm[0]))
    for number, branch in enumerate(pool.branches, 1):
        branch.instance_id = f"branch_{number:03d}"
    pool.rejections["strict_only_added"] = added
    pool.timings["strict_s"] = strict.timings["total_s"]
    pool.timings["total_s"] = round(pool.timings["total_s"] + strict.timings["total_s"], 3)
    return pool


def detect(
    image: sitk.Image, mask: sitk.Image, config: DetectorConfig | None = None
) -> Detection:
    """Run the four stages: normalize the scan, enhance vessel evidence, propose wall contacts, resolve branches."""
    config = config or DetectorConfig()
    start = perf_counter()
    validate_geometry(image, mask)
    warnings = [
        "Experimental detector: evidence scores are not calibrated probabilities.",
        "No daughter reference labels supplied; clinical accuracy is unmeasured.",
    ]
    if not np.any(sitk.GetArrayViewFromImage(mask) > 0):
        return Detection([], {"total_s": perf_counter() - start}, {}, 0, {}, warnings, config)
    scan = normalize(image, mask, config)
    warnings.extend(scan.warnings)
    prepared = perf_counter()
    evidence = enhance(scan, config)
    candidates, limit_warning = propose(scan, evidence, config)
    warnings.extend(limit_warning)
    enhanced = perf_counter()
    branches, rejections = resolve(scan, evidence, candidates, config)
    return Detection(
        branches,
        {
            "prepare_s": round(prepared - start, 3),
            "enhance_s": round(enhanced - prepared, 3),
            "trace_s": round(perf_counter() - enhanced, 3),
            "total_s": round(perf_counter() - start, 3),
        },
        scan.blood, len(candidates), rejections, warnings, config,
    )


# ----------------------------------------------------------------------------------------------
# Stage 1 · normalization: put the scan on a common grid and calibrate it to this patient's blood.
# Nothing in this stage decides anything about branches.
# ----------------------------------------------------------------------------------------------


@dataclass
class NormalizedScan:
    grid: sitk.Image
    parent: npt.NDArray[np.bool_]
    smooth: npt.NDArray
    outside: npt.NDArray
    inside: npt.NDArray
    blood: dict[str, float]
    native_spacing_mm: float
    warnings: list[str] = field(default_factory=list)


def normalize(image: sitk.Image, mask: sitk.Image, config: DetectorConfig) -> NormalizedScan:
    grid, parent = prepare_roi(image, mask, config)
    if not parent.any():
        raise ValueError("Aorta mask vanished on the working grid; use finer spacing.")
    ct = sitk.GetArrayFromImage(grid)
    spacing = config.spacing_mm
    outside = ndi.distance_transform_edt(~parent, sampling=spacing)
    inside = ndi.distance_transform_edt(parent, sampling=spacing)
    smooth = ndi.gaussian_filter(ct, sigma=0.6 / spacing)
    blood, warnings = blood_window(image, smooth, parent, outside, inside, config)
    return NormalizedScan(grid, parent, smooth, outside, inside, blood, float(max(image.GetSpacing())), warnings)


def blood_window(
    image: sitk.Image, smooth: npt.NDArray, parent: npt.NDArray, outside: npt.NDArray,
    inside: npt.NDArray, config: DetectorConfig,
) -> tuple[dict[str, float], list[str]]:
    """Per-patient HU window for contrast-filled blood, measured from the supplied aorta and its surroundings."""
    warnings: list[str] = []
    core = smooth[inside >= 2]
    if len(core) < 20:
        core = smooth[parent]
    median = float(np.median(core))
    mad = float(np.median(np.abs(core - median)) * 1.4826)
    lower = max(30.0, median - config.blood_lower_scale * max(65.0, 2.5 * mad))
    background = smooth[(outside >= 8) & (outside <= 16)]
    background_median = float(np.median(background)) if len(background) else median
    background_mad = float(np.median(np.abs(background - background_median)) * 1.4826) if len(background) else 0.0
    fraction = config.support_contrast_fraction
    if config.native_contrast_scale:
        radius_squared = config.minimum_radius_mm**2
        retention = radius_squared / (radius_squared + (0.6 * max(image.GetSpacing()))**2)
        fraction = min(fraction, config.native_contrast_scale * retention)
    if background_median < median:
        partial_volume_level = max(30.0, background_median + fraction * (median - background_median))
        if config.native_contrast_scale:
            lower = partial_volume_level
        else:
            lower = max(min(lower, partial_volume_level), (median + background_median) / 2)
    upper = median + max(120.0, 3.5 * mad)
    if median < 120:
        warnings.append("Low parent contrast: soft tissue and veins may mimic daughter arteries.")
    contrast_to_background_mad = (median - background_median) / max(background_mad, 1.0)
    if contrast_to_background_mad <= 1:
        warnings.append("Parent/background intensities overlap: inspect CT for tissue mimicking branches.")
    blood = {
        "median_hu": median, "mad_hu": mad, "lower_hu": lower, "upper_hu": upper,
        "background_median_hu": background_median, "background_mad_hu": background_mad,
        "support_fraction": fraction, "contrast_to_background_mad": contrast_to_background_mad,
    }
    return blood, warnings


# ----------------------------------------------------------------------------------------------
# Stage 2 · enhancement: per-voxel vessel evidence derived from the normalized scan.
# Still no decisions; these are maps the later stages read.
# ----------------------------------------------------------------------------------------------


@dataclass
class VesselEvidence:
    tubular: npt.NDArray
    support: npt.NDArray[np.bool_]
    radius: npt.NDArray
    junctions: FloatArray
    signed_distance: npt.NDArray
    excluded: npt.NDArray[np.bool_]
    context: TraceContext


def enhance(scan: NormalizedScan, config: DetectorConfig) -> VesselEvidence:
    spacing = config.spacing_mm
    smooth, parent, outside = scan.smooth, scan.parent, scan.outside
    lower, upper = scan.blood["lower_hu"], scan.blood["upper_hu"]
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
    signed_distance = outside - scan.inside
    excluded = cap_mask(parent, config)
    bone = smooth >= 600
    bone_distance = (
        ndi.distance_transform_edt(~bone, sampling=spacing) if bone.any() else np.full(smooth.shape, 50.0)
    ).astype(np.float32)
    voxels = np.argwhere(parent)
    sample = voxels[:: max(1, len(voxels) // 20000)].astype(float)
    _, _, vectors = np.linalg.svd(sample - sample.mean(axis=0), full_matrices=False)
    projections = sample @ vectors[0]
    median, background_median = scan.blood["median_hu"], scan.blood["background_median_hu"]
    context = TraceContext(
        smooth, (median + background_median) / 2, bone_distance, vectors[0],
        (float(projections.min()), float(projections.max())), scan.native_spacing_mm,
        median, background_median,
    )
    return VesselEvidence(tubular, support, radius, junctions, signed_distance, excluded, context)


# ----------------------------------------------------------------------------------------------
# Stage 3 · proposal: wall contacts that might be openings. A candidate is a root voxel and a blob volume.
# ----------------------------------------------------------------------------------------------


Candidate = tuple[float, npt.NDArray[np.int64], float]


def propose(scan: NormalizedScan, evidence: VesselEvidence, config: DetectorConfig) -> tuple[list[Candidate], list[str]]:
    spacing = config.spacing_mm
    outside, tubular, radius = scan.outside, evidence.tubular, evidence.radius
    shell = (
        evidence.support & ~evidence.excluded
        & (outside >= config.shell_inner_mm) & (outside <= config.shell_outer_mm)
        & (tubular >= config.shell_vesselness_floor)
    )
    labels, _ = ndi.label(shell, structure=np.ones((3, 3, 3), dtype=int))
    candidates: list[Candidate] = []
    for label_id, region in enumerate(ndi.find_objects(labels), 1):
        if region is None:
            continue
        points = np.argwhere(labels[region] == label_id)
        points += np.asarray([s.start for s in region])
        if len(points) * spacing**3 < np.pi * config.minimum_radius_mm**2 * spacing:
            continue
        quality = radius[tuple(points.T)] * (0.3 + tubular[tuple(points.T)])
        # A root deep in a wide shell leaves no room to trace outward, so prefer the wall-side voxels.
        near_wall = outside[tuple(points.T)] <= config.root_depth_mm
        if near_wall.any():
            quality = np.where(near_wall, quality, -np.inf)
        roots: list[npt.NDArray[np.int64]] = []
        for index in np.argsort(-quality, kind="stable"):
            if len(roots) >= config.roots_per_contact or not np.isfinite(quality[index]):
                break
            # Merged contact blobs hide neighbouring ostia, so allow several well-separated roots per blob.
            if all(np.linalg.norm((points[index] - r) * spacing) >= 4.0 for r in roots):
                roots.append(points[index])
                candidates.append((float(quality[index]), points[index], len(points) * spacing**3))
    candidates.sort(key=lambda c: (-c[0], tuple(c[1])))
    warnings = []
    if len(candidates) > config.maximum_candidates:
        warnings.append(f"Candidate limit reached ({config.maximum_candidates}); weaker contacts were omitted.")
    return candidates, warnings


# ----------------------------------------------------------------------------------------------
# Stage 4 · resolution: trace each candidate, keep the ones that behave like a direct daughter, measure them.
# ----------------------------------------------------------------------------------------------


def on_path(branch: Branch, other: Branch, spacing: float) -> bool:
    """True when this branch's opening, or its first lumen point, sits on the other branch's proximal path."""
    path = np.asarray(other.path_xyz_mm[1:], dtype=float)
    proximal = np.asarray(branch.path_xyz_mm[:2], dtype=float)
    if not len(path) or not len(proximal):
        return False
    distances = np.linalg.norm(path[:, None, :] - proximal[None, :, :], axis=2)
    return bool(distances.min() < 1.5 * spacing)


def resolve(
    scan: NormalizedScan, evidence: VesselEvidence, candidates: list[Candidate], config: DetectorConfig,
) -> tuple[list[Branch], dict[str, int]]:
    branches: list[Branch] = []
    rejections: dict[str, int] = {}
    for _, root, volume in candidates[:config.maximum_candidates]:
        if volume > config.broad_contact_mm3:
            reason = "broad_wall_contact"
            branch = None
        else:
            branch, reason = _trace(
                root, scan.outside, scan.parent, evidence.support, evidence.radius, evidence.tubular,
                evidence.junctions, evidence.signed_distance, evidence.context, volume, scan.grid, config,
            )
        if branch is None:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue
        reason = ""
        for old in branches:
            if np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - old.ostium_xyz_mm) >= 2.5:
                continue
            if np.linalg.norm(np.asarray(branch.seed_xyz_mm) - old.seed_xyz_mm) < 3:
                reason = "same_opening_and_path"
                break
            # One opening whose trunk forks before the seed is still one daughter (challenge doc, common trunk).
            if shares_prefix(np.asarray(branch.path_xyz_mm), np.asarray(old.path_xyz_mm)):
                reason = "common_trunk"
                break
        if reason:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue
        if config.parallel_clearance_mm > 0:
            # An opening that lies on another daughter's proximal path is the same vessel seen further along it.
            if any(on_path(branch, old, config.spacing_mm) for old in branches):
                rejections["opening_on_another_path"] = rejections.get("opening_on_another_path", 0) + 1
                continue
            downstream = [old for old in branches if on_path(old, branch, config.spacing_mm)]
            for old in downstream:
                branches.remove(old)
                rejections["opening_on_another_path"] = rejections.get("opening_on_another_path", 0) + 1
        branches.append(branch)
    branches.sort(key=lambda b: (b.ostium_xyz_mm[2], b.ostium_xyz_mm[1], b.ostium_xyz_mm[0]))
    for number, branch in enumerate(branches, 1):
        branch.instance_id = f"branch_{number:03d}"
    return branches, rejections
