import assert from "node:assert/strict";
import { test } from "node:test";
import {
  viridis,
  evidenceColor,
  viridisGradientCSS,
  UNSCORED_COLOR,
} from "../.test-dist/types.js";

test("viridis ramp endpoints and monotonic luma between them", () => {
  assert.equal(viridis(0), "#440154");
  assert.equal(viridis(1), "#fde725");
  assert.equal(viridis(-5), viridis(0), "clamps below range");
  assert.equal(viridis(5), viridis(1), "clamps above range");
  const luma = (hex) => {
    const n = parseInt(hex.slice(1), 16);
    const r = (n >> 16) & 255,
      g = (n >> 8) & 255,
      b = n & 255;
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const samples = Array.from({ length: 11 }, (_, i) => viridis(i / 10));
  for (let i = 1; i < samples.length; i++)
    assert.ok(
      luma(samples[i]) >= luma(samples[i - 1]) - 1,
      `luma should not decrease going from ${samples[i - 1]} to ${samples[i]}`,
    );
});

test("evidenceColor maps known scores and falls back to grey when unscored", () => {
  assert.equal(evidenceColor(0.98), viridis(0.98));
  assert.equal(evidenceColor(0.4), viridis(0.4));
  assert.equal(evidenceColor(undefined), UNSCORED_COLOR);
  assert.equal(evidenceColor(null), UNSCORED_COLOR);
  assert.equal(evidenceColor(NaN), UNSCORED_COLOR);
});

test("legend gradient CSS spans low to high evidence", () => {
  const css = viridisGradientCSS();
  assert.match(css, /^linear-gradient\(90deg, #440154 0%.*#fde725 100%\)$/);
});
