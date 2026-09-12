import type { Branch } from "./types";

export const FEATURE_NAMES = [
  "radius_mm",
  "mean_vesselness",
  "evidence_score",
  "path_length_mm",
  "seed_distance_mm",
  "tortuosity",
  "path_hu_relative",
  "bone_distance_mm",
  "parent_angle_degrees",
  "arc_position",
  "native_spacing_mm",
  "connector_gap",
  "candidate_volume_mm3",
];
export type ReviewLabel = "confirmed" | "rejected";
export interface Review {
  case_id: string;
  instance_id: string;
  label: ReviewLabel;
  features: number[];
  fingerprint: string;
  reviewed_at: string;
}

// The Python detector is the single source of feature values; the page only forwards them.
export function features(branch: Branch): number[] {
  const vector = branch.feature_vector;
  if (
    !Array.isArray(vector) ||
    vector.length !== FEATURE_NAMES.length ||
    !vector.every((n) => typeof n === "number" && Number.isFinite(n))
  )
    throw new Error(
      "Candidate features do not match the review contract; rebuild the frontend and restart the server.",
    );
  return [...vector];
}

function fingerprint(branch: Branch) {
  return JSON.stringify([
    branch.ostium_xyz_mm,
    branch.seed_xyz_mm,
    branch.direction_xyz,
    branch.radius_mm,
    features(branch),
  ]);
}

function isReview(value: unknown): value is Review {
  return (
    typeof value === "object" &&
    value !== null &&
    "case_id" in value &&
    typeof value.case_id === "string" &&
    "instance_id" in value &&
    typeof value.instance_id === "string" &&
    "fingerprint" in value &&
    typeof value.fingerprint === "string" &&
    "reviewed_at" in value &&
    typeof value.reviewed_at === "string" &&
    "label" in value &&
    ["confirmed", "rejected"].includes(String(value.label)) &&
    "features" in value &&
    Array.isArray(value.features) &&
    value.features.length === FEATURE_NAMES.length &&
    value.features.every(
      (n: unknown) => typeof n === "number" && Number.isFinite(n),
    )
  );
}

export class ReviewStore {
  private rows: Review[] = [];
  private key = "branchseed.reviews.v1";
  constructor() {
    try {
      const saved: unknown = JSON.parse(localStorage.getItem(this.key) || "[]");
      if (Array.isArray(saved)) this.rows = saved.filter(isReview);
    } catch {
      // Reviews remain available in memory when browser storage is unavailable.
    }
  }
  status(caseId: string, branch: Branch) {
    return (
      this.rows.find(
        (row) =>
          row.case_id === caseId &&
          row.instance_id === branch.instance_id &&
          row.fingerprint === fingerprint(branch),
      )?.label || "unreviewed"
    );
  }
  reconcile(caseId: string, branches: Branch[]) {
    const current = new Map(
      branches.map((branch) => [branch.instance_id, fingerprint(branch)]),
    );
    const before = this.rows.length;
    this.rows = this.rows.filter(
      (row) =>
        row.case_id !== caseId ||
        current.get(row.instance_id) === row.fingerprint,
    );
    const removed = before - this.rows.length;
    if (removed) this.persist();
    return removed;
  }
  set(caseId: string, branch: Branch, label: ReviewLabel | "unreviewed") {
    this.rows = this.rows.filter(
      (row) => row.case_id !== caseId || row.instance_id !== branch.instance_id,
    );
    if (label !== "unreviewed")
      this.rows.push({
        case_id: caseId,
        instance_id: branch.instance_id,
        label,
        features: features(branch),
        fingerprint: fingerprint(branch),
        reviewed_at: new Date().toISOString(),
      });
    return this.persist();
  }
  private persist() {
    try {
      localStorage.setItem(this.key, JSON.stringify(this.rows));
      return true;
    } catch {
      return false;
    }
  }
  export() {
    return {
      schema_version: 1,
      feature_names: FEATURE_NAMES,
      scope: "candidate_reviews_only",
      records: this.rows,
    };
  }
}
