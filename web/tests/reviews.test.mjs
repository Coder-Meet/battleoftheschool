import assert from "node:assert/strict";
import { test, beforeEach } from "node:test";
import { FEATURE_NAMES, ReviewStore, features } from "../.test-dist/reviews.js";

const branch = {
  instance_id: "branch_001", parent_instance_id: "aorta",
  ostium_xyz_mm: [1, 2, 3], seed_xyz_mm: [6, 2, 3], radius_mm: 2,
  direction_xyz: [1, 0, 0], path_xyz_mm: [[1, 2, 3], [6, 2, 3], [11, 2, 3]],
  evidence_score: 0.9, mean_vesselness: 0.8, warnings: [],
  features: { path_hu_relative: 0.9, bone_distance_mm: 12, parent_angle_degrees: 80, arc_position: 0.5,
    native_spacing_mm: 0.8, connector_gap: 0, candidate_volume_mm3: 60 },
  feature_vector: [2, 0.8, 0.9, 10, 5, 1, 0.9, 12, 80, 0.5, 0.8, 0, 60],
};

beforeEach(() => {
  const stored = new Map();
  globalThis.localStorage = {
    getItem: (key) => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, value),
  };
});

test("export feature order matches the Python training contract", () => {
  assert.deepEqual(FEATURE_NAMES, [
    "radius_mm", "mean_vesselness", "evidence_score", "path_length_mm", "seed_distance_mm", "tortuosity",
    "path_hu_relative", "bone_distance_mm", "parent_angle_degrees", "arc_position",
    "native_spacing_mm", "connector_gap", "candidate_volume_mm3",
  ]);
  assert.deepEqual(features(branch), branch.feature_vector);
  assert.throws(() => features({ ...branch, feature_vector: [1, 2, 3] }), /review contract/);
});

test("reviews survive reload, remain per-case and never edit predictions", () => {
  const original = structuredClone(branch);
  const store = new ReviewStore();
  store.set("case1", branch, "confirmed");
  store.set("case2", branch, "rejected");
  const reloaded = new ReviewStore();
  assert.equal(reloaded.status("case1", branch), "confirmed");
  assert.equal(reloaded.status("case2", branch), "rejected");
  assert.equal(reloaded.export().records.length, 2);
  assert.deepEqual(branch, original);
  reloaded.set("case1", branch, "unreviewed");
  assert.equal(new ReviewStore().status("case1", branch), "unreviewed");
});

test("changed measurements invalidate a stale review", () => {
  const store = new ReviewStore();
  store.set("case1", branch, "confirmed");
  const changed = { ...branch, radius_mm: 3 };
  assert.equal(store.status("case1", changed), "unreviewed");
  store.set("case1", changed, "rejected");
  assert.equal(store.export().records.length, 1);
  assert.equal(store.status("case1", changed), "rejected");
});

test("storage corruption or quota failure never prevents in-memory review export", () => {
  localStorage.setItem("branchseed.reviews.v1", "invalid JSON");
  const store = new ReviewStore();
  localStorage.setItem = () => { throw new Error("quota"); };
  assert.equal(store.set("case1", branch, "confirmed"), false);
  assert.equal(store.export().records[0].label, "confirmed");
  assert.equal(store.status("case1", branch), "confirmed");
});

test("malformed saved records are ignored", () => {
  localStorage.setItem("branchseed.reviews.v1", JSON.stringify([null, { label: "confirmed" }, 1]));
  assert.deepEqual(new ReviewStore().export().records, []);
});

test("reconciliation removes stale or disappeared candidates from training export", () => {
  const store = new ReviewStore();
  store.set("case1", branch, "confirmed");
  store.set("case1", { ...branch, instance_id: "branch_002" }, "rejected");
  store.set("case2", branch, "confirmed");
  assert.equal(store.reconcile("case1", [{ ...branch, radius_mm: 3 }]), 2);
  assert.deepEqual(store.export().records.map((r) => r.case_id), ["case2"]);
  assert.deepEqual(new ReviewStore().export(), store.export());
});
