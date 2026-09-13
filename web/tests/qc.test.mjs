import assert from "node:assert/strict";
import { test } from "node:test";
import {
  angleBetweenDegrees,
  distancePointToPolyline,
  directionPathAngleDegrees,
  selfCheckWarnings,
} from "../.test-dist/qc.js";

const baseBranch = {
  instance_id: "branch_001",
  parent_instance_id: "aorta",
  ostium_xyz_mm: [0, 0, 0],
  seed_xyz_mm: [5, 0, 0],
  radius_mm: 3,
  direction_xyz: [1, 0, 0],
  path_xyz_mm: [
    [0, 0, 0],
    [5, 0, 0],
    [10, 0, 0],
  ],
  evidence_score: 0.9,
  mean_vesselness: 0.8,
  warnings: [],
  features: {},
  feature_vector: [],
};

test("angleBetweenDegrees: parallel is 0, perpendicular is 90, opposite is 180", () => {
  assert.equal(angleBetweenDegrees([1, 0, 0], [1, 0, 0]), 0);
  assert.equal(Math.round(angleBetweenDegrees([1, 0, 0], [0, 1, 0])), 90);
  assert.equal(Math.round(angleBetweenDegrees([1, 0, 0], [-1, 0, 0])), 180);
});

test("distancePointToPolyline finds the true perpendicular distance to the nearest segment", () => {
  const polyline = [
    [0, 0, 0],
    [10, 0, 0],
  ];
  assert.equal(distancePointToPolyline([5, 3, 0], polyline), 3);
  assert.equal(distancePointToPolyline([0, 0, 0], polyline), 0);
  assert.equal(distancePointToPolyline([13, 4, 0], polyline), 5, "clamps to the nearest endpoint beyond the segment");
});

test("directionPathAngleDegrees compares direction to the full path chord, not just ostium-to-seed", () => {
  const straight = directionPathAngleDegrees(
    [
      [0, 0, 0],
      [5, 0, 0],
      [10, 0, 0],
    ],
    [1, 0, 0],
  );
  assert.equal(straight, 0);
  const bent = directionPathAngleDegrees(
    [
      [0, 0, 0],
      [5, 0, 0],
      [5, 10, 0],
    ],
    [1, 0, 0],
  );
  assert.ok(bent > 30, "a path that bends away from the reported direction should show a real divergence");
});

test("selfCheckWarnings is silent for a fully consistent branch", () => {
  assert.deepEqual(selfCheckWarnings(baseBranch), []);
});

test("selfCheckWarnings flags a seed that isn't actually on its own traced path", () => {
  const branch = { ...baseBranch, seed_xyz_mm: [5, 10, 0] };
  const warnings = selfCheckWarnings(branch);
  assert.ok(warnings.some((w) => w.includes("off its own traced path")));
});

test("selfCheckWarnings flags a direction that diverges from the overall path", () => {
  const branch = {
    ...baseBranch,
    path_xyz_mm: [
      [0, 0, 0],
      [0, 10, 0],
    ],
    direction_xyz: [1, 0, 0],
  };
  const warnings = selfCheckWarnings(branch);
  assert.ok(warnings.some((w) => w.includes("diverges")));
});

test("selfCheckWarnings flags a radius near the accept/reject boundary", () => {
  assert.ok(selfCheckWarnings({ ...baseBranch, radius_mm: 0.8 }).some((w) => w.includes("boundary")));
  assert.ok(selfCheckWarnings({ ...baseBranch, radius_mm: 7.9 }).some((w) => w.includes("boundary")));
  assert.deepEqual(selfCheckWarnings({ ...baseBranch, radius_mm: 3 }), []);
});
