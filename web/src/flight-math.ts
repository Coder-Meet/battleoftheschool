import * as THREE from "three";

export const MIN_PROGRESS = 0.02;
export const MAX_PROGRESS = 0.98;
export const ARROW_SPEED = 0.12;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Maps a flight-curve progress (0-1) to the nearest wall-diameter ring index. */
export function ringIndexAt(progress: number, ringCount: number): number {
  return Math.round(clamp(progress, 0, 1) * (ringCount - 1));
}

/** Picks a starting up-vector perpendicular to tangent, biased toward world-up. */
export function seedFlightUp(tangent: THREE.Vector3): THREE.Vector3 {
  const reference =
    Math.abs(tangent.dot(new THREE.Vector3(0, 1, 0))) > 0.9
      ? new THREE.Vector3(0, 0, 1)
      : new THREE.Vector3(0, 1, 0);
  const right = new THREE.Vector3()
    .crossVectors(tangent, reference)
    .normalize();
  return new THREE.Vector3().crossVectors(right, tangent).normalize();
}

/**
 * Parallel-transports the previous frame's up-vector onto the plane
 * perpendicular to the new tangent, instead of recomputing "up" fresh from
 * a fixed world axis every frame. A fixed-world-axis lookAt flips/rolls the
 * camera whenever the path tangent swings past vertical (exactly what the
 * aortic arch does); transporting the previous up keeps rotation continuous.
 */
export function transportFlightUp(
  tangent: THREE.Vector3,
  previousUp: THREE.Vector3,
): THREE.Vector3 {
  const right = new THREE.Vector3().crossVectors(tangent, previousUp);
  if (right.lengthSq() < 1e-6) {
    right.crossVectors(
      tangent,
      Math.abs(tangent.y) < 0.9
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(1, 0, 0),
    );
  }
  right.normalize();
  return new THREE.Vector3().crossVectors(right, tangent).normalize();
}

/** Index and distance of the sample nearest to a target point. */
export function nearestSampleIndex(
  samples: THREE.Vector3[],
  target: THREE.Vector3,
): { index: number; distance: number } {
  let bestIndex = 0;
  let bestDistanceSq = Infinity;
  samples.forEach((point, i) => {
    const distanceSq = point.distanceToSquared(target);
    if (distanceSq < bestDistanceSq) {
      bestDistanceSq = distanceSq;
      bestIndex = i;
    }
  });
  return { index: bestIndex, distance: Math.sqrt(bestDistanceSq) };
}

/**
 * Closest point to `target` on the polyline through `samples`, as a
 * continuous position (segment index + fraction along it) rather than
 * "which single sample is nearest." Snapping to the nearest discrete sample
 * point creates a Voronoi-cell boundary around each sample that doesn't
 * follow the vessel's true perpendicular cross-section wherever the
 * centreline curves (the aortic arch especially) — vertices on the outside
 * of a bend can snap to a different, non-adjacent sample than vertices
 * right next to them on the inside of the bend, tearing what should be one
 * clean ring into a jagged boundary. Projecting onto each segment instead
 * finds the true nearest point along the whole curve.
 */
export function closestPolylinePosition(
  samples: THREE.Vector3[],
  target: THREE.Vector3,
): { position: number; distance: number } {
  let bestPosition = 0;
  let bestDistanceSq = Infinity;
  const segment = new THREE.Vector3();
  const toTarget = new THREE.Vector3();
  const closest = new THREE.Vector3();
  for (let i = 0; i < samples.length - 1; i++) {
    const a = samples[i];
    const b = samples[i + 1];
    segment.subVectors(b, a);
    const lengthSq = segment.lengthSq();
    const t =
      lengthSq > 1e-12
        ? clamp(toTarget.subVectors(target, a).dot(segment) / lengthSq, 0, 1)
        : 0;
    closest.copy(a).addScaledVector(segment, t);
    const distanceSq = closest.distanceToSquared(target);
    if (distanceSq < bestDistanceSq) {
      bestDistanceSq = distanceSq;
      bestPosition = i + t;
    }
  }
  return { position: bestPosition, distance: Math.sqrt(bestDistanceSq) };
}

/** Nearest sample's arc-length parameter to a target point, for mapping an ostium onto the flight curve. */
export function progressForPoint(
  samples: THREE.Vector3[],
  target: THREE.Vector3,
): number {
  return nearestSampleIndex(samples, target).index / (samples.length - 1);
}

/** Linear-interpolated percentile (p in [0,1]) of a sample of values. */
export function percentile(values: ArrayLike<number>, p: number): number {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const index = clamp(p, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

/**
 * A [low, high] range trimmed to the given percentiles, robust to a small
 * number of outlier values (e.g. mesh vertices on a daughter-branch stub
 * fused into the parent surface, which sit much farther from the parent
 * centreline than the aorta wall itself and would otherwise blow out the
 * "widest point" reference for everyone else).
 */
export function percentileRange(
  values: ArrayLike<number>,
  low: number,
  high: number,
): [number, number] {
  return [percentile(values, low), percentile(values, high)];
}

/** All ring indices within `window` of any of `centers`, clamped to [0, ringCount). */
export function ringsNear(
  centers: number[],
  window: number,
  ringCount: number,
): Set<number> {
  const near = new Set<number>();
  for (const center of centers) {
    for (let offset = -window; offset <= window; offset++) {
      const index = Math.round(center) + offset;
      if (index >= 0 && index < ringCount) near.add(index);
    }
  }
  return near;
}

/**
 * percentileRange, but computed only over values whose index is not in
 * `excluded` — for excluding rings known to sit at a branch takeoff (where
 * a daughter's stub is fused into the parent surface) from the "widest
 * point" reference, rather than relying on a blunt percentile trim alone.
 * Falls back to the full set if excluding leaves nothing to measure.
 */
export function percentileRangeExcluding(
  values: ArrayLike<number>,
  excluded: Set<number>,
  low: number,
  high: number,
): [number, number] {
  const filtered = Array.from(values).filter((_, i) => !excluded.has(i));
  return percentileRange(filtered.length ? filtered : values, low, high);
}

/**
 * One radius per centreline sample ("ring"): the median distance of every
 * vertex nearest to that sample, instead of each vertex keeping its own
 * individually measured distance. A single vertex's nearest-centreline
 * distance is noisy (mesh irregularities, a stray branch-stub vertex on an
 * otherwise normal ring), so coloring per vertex makes the surface look
 * speckled; every vertex on the same ring instead shares one color. Rings
 * with no assigned vertices fall back to the nearest populated neighbor.
 */
export function ringRadii(
  assignments: { index: number; distance: number }[],
  sampleCount: number,
): number[] {
  const buckets: number[][] = Array.from({ length: sampleCount }, () => []);
  for (const { index, distance } of assignments) buckets[index].push(distance);
  const medians: (number | undefined)[] = buckets.map((values) => {
    if (values.length === 0) return undefined;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) / 2;
  });
  return medians.map((value, i) => {
    if (value !== undefined) return value;
    let left = i - 1;
    let right = i + 1;
    while (left >= 0 && medians[left] === undefined) left--;
    while (right < medians.length && medians[right] === undefined) right++;
    const leftValue = left >= 0 ? medians[left] : undefined;
    const rightValue = right < medians.length ? medians[right] : undefined;
    if (leftValue !== undefined && rightValue !== undefined)
      return (leftValue + rightValue) / 2;
    return leftValue ?? rightValue ?? 0;
  });
}

/** Next/previous branch progress strictly ahead of/behind `current`, or undefined at either end. */
export function nextBranchProgress(
  progresses: number[],
  current: number,
  direction: 1 | -1,
): number | undefined {
  const sorted = [...progresses].sort((a, b) => a - b);
  return direction > 0
    ? sorted.find((p) => p > current + 1e-4)
    : [...sorted].reverse().find((p) => p < current - 1e-4);
}
