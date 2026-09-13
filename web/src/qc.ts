import type { Branch, Point } from "./types";

/**
 * Self-consistency checks the viewer can run on the detector's own output,
 * with no ground truth required. They target the three things the
 * Branchseed challenge's "daughter-instance quality" score (15%) explicitly
 * grades: whether the seed lies on the matched daughter, whether the
 * predicted direction follows its proximal path, and whether the radius is
 * consistent with the daughter lumen. These are heuristics for catching
 * likely bugs before submission, not a re-implementation of the grader.
 */
const DIRECTION_ANGLE_THRESHOLD_DEG = 35;
const SEED_DISTANCE_THRESHOLD_MM = 1.5;
const RADIUS_MIN_PLAUSIBLE_MM = 1.5;
const RADIUS_MAX_PLAUSIBLE_MM = 7;

function subtract(a: Point, b: Point): Point {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function length(v: Point): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
function normalize(v: Point): Point {
  const len = length(v) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}
function dot(a: Point, b: Point): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function angleBetweenDegrees(a: Point, b: Point): number {
  const cosine = Math.min(1, Math.max(-1, dot(normalize(a), normalize(b))));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function distancePointToSegment(point: Point, a: Point, b: Point): number {
  const ab = subtract(b, a);
  const lengthSq = dot(ab, ab);
  const t =
    lengthSq > 1e-9
      ? Math.min(1, Math.max(0, dot(subtract(point, a), ab) / lengthSq))
      : 0;
  const closest: Point = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
  return length(subtract(point, closest));
}

export function distancePointToPolyline(
  point: Point,
  polyline: Point[],
): number {
  if (polyline.length === 0) return Infinity;
  if (polyline.length === 1) return length(subtract(point, polyline[0]));
  let best = Infinity;
  for (let i = 0; i < polyline.length - 1; i++)
    best = Math.min(
      best,
      distancePointToSegment(point, polyline[i], polyline[i + 1]),
    );
  return best;
}

/**
 * Angle between the reported direction and the chord across the whole
 * traced path (not just ostium-to-seed, which direction is computed from
 * and would always trivially match). A large divergence between the
 * initial direction and where the fuller trace actually goes is a real,
 * independent inconsistency signal — though some divergence is expected
 * from genuine vessel curvature, so the threshold is generous.
 */
export function directionPathAngleDegrees(
  path: Point[],
  direction: Point,
): number | undefined {
  if (path.length < 2) return undefined;
  const chord = subtract(path[path.length - 1], path[0]);
  if (length(chord) < 1e-6) return undefined;
  return angleBetweenDegrees(chord, direction);
}

export function selfCheckWarnings(branch: Branch): string[] {
  const warnings: string[] = [];
  const angle = directionPathAngleDegrees(
    branch.path_xyz_mm,
    branch.direction_xyz,
  );
  if (angle !== undefined && angle > DIRECTION_ANGLE_THRESHOLD_DEG) {
    warnings.push(
      `Self-check: direction diverges ${angle.toFixed(0)}° from the overall traced path.`,
    );
  }
  const seedDistance = distancePointToPolyline(
    branch.seed_xyz_mm,
    branch.path_xyz_mm,
  );
  if (seedDistance > SEED_DISTANCE_THRESHOLD_MM) {
    warnings.push(
      `Self-check: seed sits ${seedDistance.toFixed(1)}mm off its own traced path.`,
    );
  }
  if (
    branch.radius_mm < RADIUS_MIN_PLAUSIBLE_MM ||
    branch.radius_mm > RADIUS_MAX_PLAUSIBLE_MM
  ) {
    warnings.push(
      `Self-check: radius ${branch.radius_mm.toFixed(1)}mm is near the detector's accept/reject boundary.`,
    );
  }
  return warnings;
}
