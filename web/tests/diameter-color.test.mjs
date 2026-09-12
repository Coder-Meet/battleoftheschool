import assert from "node:assert/strict";
import { test } from "node:test";
import { diameterColor, diameterColorRGB, diameterGradientCSS } from "../.test-dist/types.js";

test("diameter ramp: thinnest is black, widest is white, clamped outside [0,1]", () => {
  assert.equal(diameterColor(0), "#000000");
  assert.equal(diameterColor(1), "#ffffff");
  assert.equal(diameterColor(-5), diameterColor(0), "clamps below range");
  assert.equal(diameterColor(5), diameterColor(1), "clamps above range");
});

test("diameterColorRGB is a grayscale ramp that brightens from thinnest to widest", () => {
  const samples = Array.from({ length: 11 }, (_, i) => diameterColorRGB(i / 10));
  for (let i = 1; i < samples.length; i++) {
    const [r, g, b] = samples[i];
    assert.ok(Math.abs(r - g) < 1e-9 && Math.abs(g - b) < 1e-9, "should stay grayscale (r=g=b)");
    const brightness = r + g + b;
    const previousBrightness = samples[i - 1].reduce((a, c) => a + c, 0);
    assert.ok(brightness >= previousBrightness - 1e-9, "should not darken as the vessel widens");
  }
});

test("legend gradient CSS spans thinnest (black) to widest (white)", () => {
  const css = diameterGradientCSS();
  assert.match(css, /^linear-gradient\(90deg, #000000 0%.*#ffffff 100%\)$/);
});
