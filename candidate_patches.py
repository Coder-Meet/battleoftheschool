"""Physical, label-free candidate patches shared by research training and inference.

Channels are XY, XZ, YZ views, each containing CT, parent, vesselness, path.
Columns/rows follow physical +X/+Y, +X/+Z, +Y/+Z respectively (not voxel
indices or radiological display conventions). Pixel ``size // 2`` in both
dimensions is the ostium, including for even sizes. Distances are in mm.

CT is clipped to [0, 1] using a scan-wide parent median and non-parent body
median (HU > -300), with a scale floor of 3 parent MADs or 1 HU. Statistics
use deterministic, evenly spaced samples within each population, at most
65536 each. Padding is zero in every output channel and excluded from features.
Vesselness is raw, clipped [0, 1] Sato tubeness after 0.6 mm smoothing, at
0.8/1.5/2.5 mm scales, without per-patch percentile scaling. These are the
detector's physical scales; detector.enhance also builds unrelated full-ROI
topology and uses a different amplitude calibration, so it is not called here.
The path channel is exp(-distance_to_polyline**2 / (2 * 0.75**2)), truncated
at 3 sigma. It describes the proposal only, never a reference segmentation.

Extra features, in EXTRA_FEATURE_NAMES order:
* ostium_local_contrast: median normalized CT within 3 mm minus that in a
  5--8 mm annulus, both outside the parent and inside the acquired image.
* patch_vesselness_std: population standard deviation in the observed 3D
  output cube, including parent voxels.
* parent_wall_curvature: magnitude of fitted mean curvature (1/mm), from a
  quadratic surface fitted to observed parent-wall crossings within 5 mm.
  Acquisition edges are excluded. Missing feature support raises ValueError.

Input validation and scan statistics stream in bounded blocks. One local
cube (12 mm halo) is resampled per candidate, never an entire CT. Local cubes
are limited to 128**3 voxels; larger size/finer spacing requests fail before
allocation. Output storage necessarily grows as N * 12 * size**2 * 4 bytes;
there is no candidate cap. Inputs and Detection are never mutated.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import sato

from detector import Branch, Detection

EXTRA_FEATURE_NAMES = ["ostium_local_contrast", "patch_vesselness_std", "parent_wall_curvature"]
CHANNEL_NAMES = [
    f"{view}_{channel}"
    for view in ("xy", "xz", "yz")
    for channel in ("ct", "parent", "vesselness", "path")
]
PREPROCESSING_VERSION = "physical-candidate-patches-v1"
_BLOCK = 262144
_SAMPLES = 65536
_HALO_MM = 12.0
_PATH_SIGMA_MM = 0.75
_SATO_SCALES_MM = (0.8, 1.5, 2.5)
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CandidateData:
    patches: npt.NDArray[np.float32]
    extra_features: FloatArray


def preprocessing_metadata(size: int = 32, spacing_mm: float = 1.0) -> dict:
    """JSON-compatible preprocessing specification; source/model hashes belong in the sidecar."""
    _grid_size(size, spacing_mm)
    return {
        "version": PREPROCESSING_VERSION,
        "size": size, "spacing_mm": float(spacing_mm),
        "channel_names": list(CHANNEL_NAMES),
        "extra_feature_names": list(EXTRA_FEATURE_NAMES),
        "physical_axes": ["xy", "xz", "yz"], "center_pixel": size // 2,
        "ct_normalization": "clip((HU-body_median)/max(abs(parent_median-body_median),3*parent_MAD,1),0,1)",
        "body_min_hu": -300.0, "mad_factor": 1.4826, "samples_per_population": _SAMPLES,
        "sampling": "evenly spaced population indices in native C-order",
        "ct_interpolation": "linear", "parent_interpolation": "nearest",
        "padding": 0.0, "halo_mm": _HALO_MM,
        "smoothing_sigma_mm": 0.6, "sato_sigmas_mm": list(_SATO_SCALES_MM),
        "sato_black_ridges": False, "sato_scaling": "raw response clipped to [0,1]",
        "path_sigma_mm": _PATH_SIGMA_MM, "path_truncation_sigma": 3.0,
        "contrast_radii_mm": [3.0, 5.0, 8.0],
        "curvature_fit_radius_mm": 5.0, "curvature_units": "1/mm",
    }


def _grid_size(size: int, spacing: float) -> tuple[int, int]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Patch size must be a positive integer.")
    if isinstance(spacing, bool) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Patch spacing must be positive and finite.")
    if spacing < 2 * _HALO_MM / 128 or size > 128:
        raise ValueError("Local patch grid exceeds the 128**3 voxel memory budget.")
    halo = int(np.ceil(_HALO_MM / spacing))
    width = size + 2 * halo
    if width > 128 or not np.isfinite(width * spacing):
        raise ValueError("Local patch grid exceeds the 128**3 voxel memory budget.")
    return halo, width


def _validate_inputs(image: sitk.Image, mask: sitk.Image) -> tuple[int, int]:
    for volume in (image, mask):
        if volume.GetDimension() != 3 or volume.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("CT and parent mask must be scalar three-dimensional images.")
        if min(volume.GetSize()) < 1:
            raise ValueError("CT and parent mask must have a nonempty grid.")
        spacing = np.asarray(volume.GetSpacing())
        direction = np.asarray(volume.GetDirection()).reshape(3, 3)
        if not (
            np.isfinite(spacing).all() and np.all(spacing > 0)
            and np.isfinite(volume.GetOrigin()).all() and np.isfinite(direction).all()
        ):
            raise ValueError("Image geometry must be finite with positive spacing.")
        if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-4):
            raise ValueError("Image direction must be orthonormal.")
        with np.errstate(over="ignore", invalid="ignore"):
            extent = np.abs(direction) @ (np.asarray(volume.GetSize()) * spacing)
            representable = np.isfinite(np.abs(volume.GetOrigin()) + extent).all()
        if not representable:
            raise ValueError("Image physical extent cannot be represented safely.")
    if image.GetSize() != mask.GetSize() or any(
        not np.allclose(left, right, rtol=0, atol=1e-4)
        for left, right in (
            (image.GetSpacing(), mask.GetSpacing()), (image.GetOrigin(), mask.GetOrigin()),
            (image.GetDirection(), mask.GetDirection()),
        )
    ):
        raise ValueError("CT and parent mask must share physical geometry.")
    ct = sitk.GetArrayViewFromImage(image).reshape(-1)
    parent = sitk.GetArrayViewFromImage(mask).reshape(-1)
    if np.iscomplexobj(ct) or np.iscomplexobj(parent):
        raise ValueError("CT and mask must contain real voxel values.")
    parent_count = body_count = 0
    positive_label: float | None = None
    for start in range(0, len(ct), _BLOCK):
        values, labels = ct[start:start + _BLOCK], parent[start:start + _BLOCK]
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("Input contains nonfinite voxel values.")
        unique = np.unique(labels)
        positive = unique[unique > 0]
        if np.any(unique < 0) or len(positive) > 1:
            raise ValueError("Supply a binary parent mask.")
        if len(positive):
            label = float(positive[0])
            if positive_label is not None and label != positive_label:
                raise ValueError("Supply a binary parent mask.")
            positive_label = label
        parent_count += int(np.count_nonzero(labels > 0))
        body_count += int(np.count_nonzero((labels == 0) & (values > -300)))
    return parent_count, body_count


def _scan_window(
    image: sitk.Image, mask: sitk.Image, counts: tuple[int, int],
) -> tuple[float, float]:
    if min(counts) == 0:
        raise ValueError("Scan normalization requires parent and non-parent body voxels (HU > -300).")
    targets = [
        np.linspace(0, count - 1, min(count, _SAMPLES), dtype=np.int64) for count in counts
    ]
    samples = [np.empty(len(indices), dtype=np.float64) for indices in targets]
    seen = [0, 0]
    ct = sitk.GetArrayViewFromImage(image).reshape(-1)
    parent = sitk.GetArrayViewFromImage(mask).reshape(-1)
    for start in range(0, len(ct), _BLOCK):
        values, labels = ct[start:start + _BLOCK], parent[start:start + _BLOCK]
        for population, selected in enumerate((labels > 0, (labels == 0) & (values > -300))):
            local = values[selected]
            left, right = np.searchsorted(targets[population], [seen[population], seen[population] + len(local)])
            samples[population][left:right] = local[targets[population][left:right] - seen[population]]
            seen[population] += len(local)
    blood = float(np.median(samples[0]))
    body = float(np.median(samples[1]))
    with np.errstate(over="ignore", invalid="ignore"):
        mad = float(np.median(np.abs(samples[0] - blood)) * 1.4826)
        scale = max(abs(blood - body), 3 * mad, 1.0)
    if not np.isfinite(scale):
        raise ValueError("CT intensity range cannot be normalized safely.")
    return body, scale


def _branch_path(branch: Branch, image: sitk.Image) -> FloatArray:
    try:
        path = np.asarray(branch.path_xyz_mm, dtype=np.float64)
        ostium = np.asarray(branch.ostium_xyz_mm, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError("Malformed branch path or ostium.") from error
    if (
        path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or ostium.shape != (3,)
        or not np.isfinite(path).all() or not np.isfinite(ostium).all()
    ):
        raise ValueError("Branch path must contain at least two finite physical XYZ points.")
    if not np.allclose(path[0], ostium, rtol=0, atol=1e-4):
        raise ValueError("Branch path must start at its ostium.")
    if not np.any(np.diff(path, axis=0)):
        raise ValueError("Branch path must have positive length.")
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        indices = ((path - image.GetOrigin()) @ np.linalg.inv(direction).T) / image.GetSpacing()
    if not np.isfinite(indices).all() or np.any(indices < -0.5) or np.any(
        indices >= np.asarray(image.GetSize()) - 0.5
    ):
        raise ValueError("Branch path lies outside the acquired image.")
    return path


def _observed(grid: sitk.Image, image: sitk.Image) -> npt.NDArray[np.bool_]:
    z, y, x = np.indices(grid.GetSize()[::-1], dtype=np.float64)
    coords = np.stack((x, y, z), axis=-1)
    coords *= grid.GetSpacing()
    coords += grid.GetOrigin()
    coords -= image.GetOrigin()
    inverse = np.linalg.inv(np.asarray(image.GetDirection()).reshape(3, 3))
    coords = (coords @ inverse.T) / image.GetSpacing()
    return np.all((coords >= -0.5) & (coords < np.asarray(image.GetSize()) - 0.5), axis=-1)


def _wall_curvature(parent: npt.NDArray[np.bool_], valid: npt.NDArray[np.bool_], center: int, spacing: float) -> float:
    crossings = []
    for axis in range(3):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(lower), tuple(upper)
        points = np.argwhere((parent[a] != parent[b]) & valid[a] & valid[b]).astype(np.float64)
        points[:, axis] += 0.5
        points = (points - center) * spacing
        crossings.append(points[np.linalg.norm(points, axis=1) <= 5.0])
    wall = np.concatenate(crossings)
    if len(wall) < 6:
        raise ValueError("Insufficient observed parent wall for curvature.")
    weights = np.exp(-np.sum(wall**2, axis=1) / 18.0)
    centered = wall - np.average(wall, axis=0, weights=weights)
    _, _, axes = np.linalg.svd(centered * np.sqrt(weights[:, None]), full_matrices=False)
    u, v, height = (centered @ axes.T).T
    design = np.column_stack((u**2, u * v, v**2, u, v, np.ones(len(u))))
    coef, _, rank, _ = np.linalg.lstsq(design * np.sqrt(weights[:, None]), height * np.sqrt(weights), rcond=None)
    if rank < 6:
        raise ValueError("Observed parent wall is degenerate for curvature.")
    a, b, c, d, e, _ = coef
    return float(abs(((1 + e**2) * a - d * e * b + (1 + d**2) * c) / (1 + d**2 + e**2)**1.5))


def _path_plane(points: FloatArray, path: FloatArray) -> npt.NDArray[np.float32]:
    distance2 = np.full(points.shape[:-1], np.inf)
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        length2 = float(segment @ segment)
        if length2 == 0:
            continue
        relative = points - start
        fraction = np.clip((relative @ segment) / length2, 0, 1)
        distance2 = np.minimum(distance2, np.sum((relative - fraction[..., None] * segment)**2, axis=-1))
    values = np.exp(-distance2 / (2 * _PATH_SIGMA_MM**2))
    values[distance2 > (3 * _PATH_SIGMA_MM)**2] = 0
    return values.astype(np.float32)


def extract_candidate_data(
    image: sitk.Image, mask: sitk.Image, result: Detection, size: int = 32, spacing_mm: float = 1.0,
) -> CandidateData:
    """Return float32 (N,12,size,size) patches and float64 (N,3) features in branch order.

    Invalid grids, nonfinite input, malformed paths, unsupported local feature
    geometry, or a local grid above the memory budget raise ValueError.
    Empty detections still validate input and return shaped empty arrays.
    """
    halo, width = _grid_size(size, spacing_mm)
    counts = _validate_inputs(image, mask)
    for branch in result.branches:
        _branch_path(branch, image)
    count = len(result.branches)
    patches = np.empty((count, 12, size, size), dtype=np.float32)
    features = np.empty((count, 3), dtype=np.float64)
    if not count:
        return CandidateData(patches, features)
    body, scale = _scan_window(image, mask, counts)
    center = halo + size // 2
    core = (slice(halo, halo + size),) * 3
    offsets = (np.arange(width, dtype=np.float64) - center) * spacing_mm
    radius2 = offsets[:, None, None]**2 + offsets[None, :, None]**2 + offsets[None, None, :]**2
    plane_offsets = (np.arange(size, dtype=np.float64) - size // 2) * spacing_mm
    columns, rows = np.meshgrid(plane_offsets, plane_offsets)
    for index, branch in enumerate(result.branches):
        path = _branch_path(branch, image)
        grid = sitk.Image([width] * 3, sitk.sitkFloat32)
        grid.SetSpacing([spacing_mm] * 3)
        grid.SetOrigin((np.asarray(branch.ostium_xyz_mm) - center * spacing_mm).tolist())
        raw = sitk.GetArrayFromImage(sitk.Resample(
            image, grid, sitk.Transform(), sitk.sitkLinear, body, sitk.sitkFloat64,
        ))
        with np.errstate(over="ignore", invalid="ignore"):
            ct = np.clip((raw - body) / scale, 0, 1).astype(np.float32)
        if not np.isfinite(ct).all():
            raise ValueError("Resampled CT cannot be normalized safely.")
        parent = sitk.GetArrayFromImage(sitk.Resample(
            mask, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkFloat64,
        )) > 0
        valid = _observed(grid, image)
        ct[~valid] = 0
        parent &= valid
        smooth = ndi.gaussian_filter(ct, 0.6 / spacing_mm, mode="constant", cval=0)
        vesselness = np.clip(sato(
            smooth, sigmas=[sigma / spacing_mm for sigma in _SATO_SCALES_MM],
            black_ridges=False, mode="constant", cval=0,
        ), 0, 1).astype(np.float32)
        vesselness[~valid] = 0
        near = valid & ~parent & (radius2 <= 3**2)
        background = valid & ~parent & (radius2 >= 5**2) & (radius2 <= 8**2)
        if not near.any() or not background.any() or not valid[core].any():
            raise ValueError("Insufficient observed non-parent tissue for local contrast.")
        features[index] = (
            float(np.median(ct[near]) - np.median(ct[background])),
            float(np.std(vesselness[core][valid[core]], dtype=np.float64)),
            _wall_curvature(parent, valid, center, spacing_mm),
        )
        for view, (horizontal, vertical) in enumerate(combinations(range(3), 2)):
            fixed_axis = 3 - horizontal - vertical
            plane: list[slice | int] = [slice(halo, halo + size)] * 3
            plane[2 - fixed_axis] = center
            selection = tuple(plane)
            points = np.zeros((size, size, 3), dtype=np.float64)
            points[..., horizontal], points[..., vertical] = columns, rows
            path_values = _path_plane(points, path - branch.ostium_xyz_mm)
            path_values[~valid[selection]] = 0
            for channel, values in enumerate((ct[selection], parent[selection], vesselness[selection], path_values)):
                patches[index, view * 4 + channel] = values
    if not np.isfinite(patches).all() or not np.isfinite(features).all():
        raise ValueError("Candidate extraction produced nonfinite data.")
    return CandidateData(patches, features)
