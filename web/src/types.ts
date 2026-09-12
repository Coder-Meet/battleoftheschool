export type Point = [number, number, number];

export interface Branch {
  instance_id: string;
  parent_instance_id: string;
  ostium_xyz_mm: Point;
  seed_xyz_mm: Point;
  radius_mm: number;
  direction_xyz: Point;
  path_xyz_mm: Point[];
  evidence_score: number;
  mean_vesselness: number;
  warnings: string[];
}

export interface Case {
  case_id: string;
  size_xyz: Point;
  origin_xyz: Point;
  basis: number[][];
  native_size_xyz: Point;
  native_spacing_xyz: Point;
  native_direction: number[];
  aorta_volume_ml: number;
  coverage_mm: number;
  mesh: { vertices: number[]; faces: number[] };
  centerline: Point[];
  branches: Branch[];
  prediction: {
    case_id: string;
    parent: { instance_id: string };
    daughters: object[];
  };
  diagnostics: {
    timings: Record<string, number>;
    blood_model: Record<string, number>;
    candidates: number;
    rejections: Record<string, number>;
    warnings: string[];
  };
}

export const COLORS = [
  "#63ddbd",
  "#eebc6d",
  "#8dabff",
  "#d998e8",
  "#6fd3ec",
  "#ff9b85",
];
export const branchName = (id: string) => `Branch ${id.split("_")[1]}`;
