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
  features: Record<string, number>;
  feature_vector: number[];
}

export interface Case {
  case_id: string;
  profile: string;
  feature_names: string[];
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

// Detection evidence color scale (viridis). Not a calibrated confidence —
// see evidence_score in the branch data, which is a heuristic 0-1 score.
const VIRIDIS_STOPS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
];
export const UNSCORED_COLOR = "#7b8796";

export function viridis(t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (VIRIDIS_STOPS.length - 1);
  const index = Math.min(VIRIDIS_STOPS.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  const [r1, g1, b1] = VIRIDIS_STOPS[index];
  const [r2, g2, b2] = VIRIDIS_STOPS[index + 1];
  const mix = (a: number, b: number) =>
    Math.round(a + (b - a) * fraction)
      .toString(16)
      .padStart(2, "0");
  return `#${mix(r1, r2)}${mix(g1, g2)}${mix(b1, b2)}`;
}

export function evidenceColor(score: number | undefined | null): string {
  return typeof score === "number" && Number.isFinite(score)
    ? viridis(score)
    : UNSCORED_COLOR;
}

export function viridisGradientCSS(steps = 10): string {
  const stops = Array.from({ length: steps }, (_, i) => {
    const t = i / (steps - 1);
    return `${viridis(t)} ${(t * 100).toFixed(0)}%`;
  });
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

// Local wall-radius color scale: t=0 is the thinnest point of the parent
// aorta, t=1 the widest. Not evidence/confidence — purely geometric.
const DIAMETER_THIN_COLOR: [number, number, number] = [0, 0, 0]; // thinnest: black
const DIAMETER_WIDE_COLOR: [number, number, number] = [255, 255, 255]; // widest: white

export function diameterColorRGB(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const [r1, g1, b1] = DIAMETER_THIN_COLOR;
  const [r2, g2, b2] = DIAMETER_WIDE_COLOR;
  return [
    (r1 + (r2 - r1) * clamped) / 255,
    (g1 + (g2 - g1) * clamped) / 255,
    (b1 + (b2 - b1) * clamped) / 255,
  ];
}

export function diameterColor(t: number): string {
  const [r, g, b] = diameterColorRGB(t).map((c) =>
    Math.round(c * 255)
      .toString(16)
      .padStart(2, "0"),
  );
  return `#${r}${g}${b}`;
}

export function diameterGradientCSS(steps = 10): string {
  const stops = Array.from({ length: steps }, (_, i) => {
    const t = i / (steps - 1);
    return `${diameterColor(t)} ${(t * 100).toFixed(0)}%`;
  });
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}
