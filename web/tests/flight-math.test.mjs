import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import {
  MIN_PROGRESS,
  MAX_PROGRESS,
  ARROW_SPEED,
  clamp,
  seedFlightUp,
  transportFlightUp,
  progressForPoint,
  nextBranchProgress,
  nearestSampleIndex,
  closestPolylinePosition,
  percentile,
  percentileRange,
  percentileRangeExcluding,
  ringRadii,
  ringIndexAt,
  ringsNear,
} from "../.test-dist/flight-math.js";
import { diameterColorRGB } from "../.test-dist/types.js";

test("clamp stays within [min, max] and range constants bound the flight slider", () => {
  assert.equal(clamp(5, 0, 1), 1);
  assert.equal(clamp(-5, 0, 1), 0);
  assert.equal(clamp(0.5, 0, 1), 0.5);
  assert.ok(MIN_PROGRESS > 0 && MAX_PROGRESS < 1);
  assert.ok(ARROW_SPEED > 0);
});

test("ringIndexAt maps flight progress to the nearest ring, clamped at both ends", () => {
  assert.equal(ringIndexAt(0, 150), 0);
  assert.equal(ringIndexAt(1, 150), 149);
  assert.equal(ringIndexAt(0.5, 151), 75);
  assert.equal(ringIndexAt(-5, 150), 0, "clamps below range");
  assert.equal(ringIndexAt(5, 150), 149, "clamps above range");
});

test("no roll/flip when the path tangent sweeps past vertical (the aortic-arch case)", () => {
  // Simulate a centreline whose tangent rotates a full 180 degrees through
  // straight-up and straight-down: exactly the case where a fixed-world-up
  // lookAt degenerates and the camera used to flip/roll.
  const steps = 180;
  const tangentAt = (i) => {
    const theta = (Math.PI * i) / steps;
    return new THREE.Vector3(Math.sin(theta), Math.cos(theta), 0).normalize();
  };
  let up = seedFlightUp(tangentAt(0));
  let previousUp = up.clone();
  for (let i = 1; i <= steps; i++) {
    const tangent = tangentAt(i);
    up = transportFlightUp(tangent, previousUp);
    assert.ok(
      Math.abs(up.length() - 1) < 1e-6,
      `up must stay unit length at step ${i}`,
    );
    assert.ok(
      Math.abs(up.dot(tangent)) < 1e-6,
      `up must stay perpendicular to tangent at step ${i}`,
    );
    const stepAngle = (Math.acos(clamp(previousUp.dot(up), -1, 1)) * 180) / Math.PI;
    assert.ok(
      stepAngle < 5,
      `up rotated ${stepAngle.toFixed(2)} degrees in one step ${i} (a flip/roll would jump ~180)`,
    );
    previousUp = up;
  }
});

test("seedFlightUp is always perpendicular to the tangent, even near world-up", () => {
  for (const tangent of [
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(1, 1, 1).normalize(),
  ]) {
    const up = seedFlightUp(tangent);
    assert.ok(Math.abs(up.dot(tangent)) < 1e-6);
    assert.ok(Math.abs(up.length() - 1) < 1e-6);
  }
});

test("progressForPoint finds the nearest sample's arc-length parameter", () => {
  const samples = Array.from(
    { length: 11 },
    (_, i) => new THREE.Vector3(i * 10, 0, 0),
  );
  assert.equal(progressForPoint(samples, new THREE.Vector3(0, 0, 0)), 0);
  assert.equal(progressForPoint(samples, new THREE.Vector3(100, 0, 0)), 1);
  assert.equal(progressForPoint(samples, new THREE.Vector3(52, 0, 0)), 0.5);
});

test("closestPolylinePosition finds a true point on a segment, not just the nearest sample vertex", () => {
  const samples = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(10, 0, 0),
    new THREE.Vector3(10, 10, 0),
  ];
  const target = new THREE.Vector3(9, 1.2, 0);
  const projected = closestPolylinePosition(samples, target);
  const nearest = nearestSampleIndex(samples, target);
  assert.ok(
    projected.distance < nearest.distance,
    "projecting onto a segment should find a point at least as close as any single sample",
  );
  assert.ok(
    projected.position > 0 && projected.position < samples.length - 1,
    "should be a continuous position along the polyline",
  );
  assert.ok(
    Math.abs(projected.position - Math.round(projected.position)) > 1e-6,
    "should not collapse to an exact sample index for an off-vertex point",
  );
});

test("closestPolylinePosition clamps to the polyline's ends", () => {
  const samples = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(10, 0, 0)];
  assert.equal(closestPolylinePosition(samples, new THREE.Vector3(-5, 3, 0)).position, 0);
  assert.equal(closestPolylinePosition(samples, new THREE.Vector3(15, 3, 0)).position, 1);
});

test("closestPolylinePosition keeps positions close across a sharp bend, unlike nearest-sample snapping", () => {
  // An L-bend. Two points straddling the corner, close together in 3D
  // space, should get positions close together too -- nearest-sample
  // snapping can instead tear them apart onto non-adjacent samples,
  // fracturing what should be one continuous ring into a jagged boundary.
  const samples = Array.from({ length: 21 }, (_, i) =>
    i <= 10 ? new THREE.Vector3(i, 0, 0) : new THREE.Vector3(10, i - 10, 0),
  );
  const justBefore = closestPolylinePosition(samples, new THREE.Vector3(9.9, 0.3, 0));
  const justAfter = closestPolylinePosition(samples, new THREE.Vector3(10.3, 0.4, 0));
  assert.ok(
    Math.abs(justAfter.position - justBefore.position) < 1,
    "two points straddling the corner should land within one ring of each other",
  );
});

test("ringRadii: every vertex on a ring shares one median radius, unmoved by a single noisy vertex", () => {
  const assignments = [
    { index: 0, distance: 10 },
    { index: 0, distance: 10 },
    { index: 0, distance: 10 },
    { index: 0, distance: 40 }, // one stray branch-stub vertex snapped to this ring
    { index: 2, distance: 20 },
  ];
  const radii = ringRadii(assignments, 4);
  assert.equal(radii.length, 4);
  assert.equal(radii[0], 10, "the outlier should not drag the ring's median");
  assert.equal(radii[2], 20);
});

test("ringRadii fills a ring with no assigned vertices from its nearest populated neighbors", () => {
  const assignments = [
    { index: 0, distance: 10 },
    { index: 2, distance: 20 },
  ];
  const radii = ringRadii(assignments, 3);
  assert.equal(radii[1], 15, "empty ring 1 should average its populated neighbors (0 and 2)");
});

test("wall-diameter mapping: a narrowed cross-section renders darker, a bulge renders lighter, uniformly per ring", () => {
  // Mirrors AortaViewer.wallDiameterColors: nearest-centreline distance per
  // vertex, grouped into one radius per ring, then mapped to thinnest/widest.
  const centerline = Array.from(
    { length: 50 },
    (_, i) => new THREE.Vector3(i * 5, 0, 0),
  );
  // Vertices ringing three cross-sections: a narrowed one (radius 4), a
  // normal one (radius 10), and a bulge (radius 16).
  const ring = (x, radius) =>
    Array.from({ length: 12 }, (_, i) => {
      const a = (2 * Math.PI * i) / 12;
      return new THREE.Vector3(x, Math.cos(a) * radius, Math.sin(a) * radius);
    });
  const vertices = [...ring(50, 4), ...ring(120, 10), ...ring(200, 16)];
  const assignments = vertices.map((v) => {
    const { position, distance } = closestPolylinePosition(centerline, v);
    return { index: clamp(Math.round(position), 0, centerline.length - 1), distance };
  });
  const radiiByRing = ringRadii(assignments, centerline.length);
  const [low, high] = percentileRange(radiiByRing, 0.02, 0.9);
  const colorAt = (radius) => diameterColorRGB((radius - low) / (high - low));
  const brightness = (c) => c[0] + c[1] + c[2];
  const narrowRing = assignments[0].index;
  const normalRing = assignments[12].index;
  const bulgeRing = assignments[24].index;
  const narrow = colorAt(radiiByRing[narrowRing]);
  const normal = colorAt(radiiByRing[normalRing]);
  const bulge = colorAt(radiiByRing[bulgeRing]);
  assert.ok(
    brightness(narrow) < brightness(normal) && brightness(normal) < brightness(bulge),
    "brightness should rise as the vessel widens (black = thinnest, white = widest)",
  );
  // Every vertex on the narrow ring gets the exact same color as every
  // other vertex on that ring, not its own individually measured shade.
  for (let i = 0; i < 12; i++) {
    assert.deepEqual(colorAt(radiiByRing[assignments[i].index]), narrow);
  }
});

test("percentile interpolates and clamps p to [0,1]", () => {
  const values = [10, 20, 30, 40, 50];
  assert.equal(percentile(values, 0), 10);
  assert.equal(percentile(values, 1), 50);
  assert.equal(percentile(values, 0.5), 30);
  assert.equal(percentile(values, -1), percentile(values, 0));
  assert.equal(percentile(values, 2), percentile(values, 1));
  assert.deepEqual(percentileRange(values, 0.2, 0.8), [
    percentile(values, 0.2),
    percentile(values, 0.8),
  ]);
});

test("percentile-trimmed wall-diameter range ignores a daughter-branch-stub outlier", () => {
  // 200 ordinary wall vertices with radius clustered around 10-12mm, plus a
  // handful of vertices on a branch stub fused into the mesh sitting far
  // outside that range (radius ~40mm). Mirrors AortaViewer.wallDiameterColors.
  const wallRadii = Array.from(
    { length: 200 },
    (_, i) => 10 + (i % 20) * 0.1,
  );
  const stubRadii = [38, 39, 40, 41, 42];
  const radii = [...wallRadii, ...stubRadii];
  const [low, high] = percentileRange(radii, 0.02, 0.9);
  assert.ok(high < 20, `trimmed high (${high}) should exclude the ~40mm stub outliers`);
  assert.ok(low >= 10 && low <= 12, `trimmed low (${low}) should stay within the normal wall range`);
  // Without trimming, the outlier would have dominated the range.
  assert.ok(Math.max(...radii) - Math.min(...radii) > (high - low) * 2);
});

test("ringsNear marks a window around each center ring, clamped to valid indices", () => {
  const near = ringsNear([5, 20], 2, 25);
  assert.deepEqual([...near].sort((a, b) => a - b), [3, 4, 5, 6, 7, 18, 19, 20, 21, 22]);
  const nearEdges = ringsNear([0, 24], 2, 25);
  assert.ok(!nearEdges.has(-1) && !nearEdges.has(25), "should clamp at both ends of the ring range");
});

test("percentileRangeExcluding uses the real vessel range instead of a blunt trim, once branch rings are excluded", () => {
  // A vessel that's genuinely wide (14-16mm) for most of its length, with
  // one branch ring (~40mm, its stub fused into the surface) excluded by
  // index rather than guessed away by percentile alone.
  const wallRadii = Array.from({ length: 100 }, (_, i) => 14 + (i % 20) * 0.1);
  const radii = [...wallRadii, 40];
  const stubRingIndex = radii.length - 1;
  const excluded = ringsNear([stubRingIndex], 1, radii.length);
  const [low, high] = percentileRangeExcluding(radii, excluded, 0.01, 0.99);
  assert.ok(high < 20, "excluding the stub ring should keep the real vessel range, not the outlier");
  assert.ok(high >= 15.5, "a light 1-99th trim on the real vessel values should preserve most of its width, unlike a 90th-percentile cut");
});

test("percentileRangeExcluding falls back to the full set if exclusion empties it", () => {
  const values = [1, 2, 3];
  const excluded = new Set([0, 1, 2]);
  assert.deepEqual(percentileRangeExcluding(values, excluded, 0, 1), percentileRange(values, 0, 1));
});

test("nextBranchProgress jumps to the nearest ostium ahead/behind, clamped at the ends", () => {
  const progresses = [0.1, 0.35, 0.6, 0.9];
  assert.equal(nextBranchProgress(progresses, 0.2, 1), 0.35);
  assert.equal(nextBranchProgress(progresses, 0.2, -1), 0.1);
  assert.equal(nextBranchProgress(progresses, 0.9, 1), undefined);
  assert.equal(nextBranchProgress(progresses, 0.1, -1), undefined);
  assert.equal(nextBranchProgress(progresses, 0.35, 1), 0.6, "starting exactly on an ostium still advances");
});
