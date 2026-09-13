> Historical snapshot. Its status, thresholds and task instructions describe an earlier revision. For current operation use [the workflow](../../PRODUCTION_WORKFLOW.md), [final handoff](../../FINAL_HANDOFF.md), and [the current plan](../../CURRENT_E2E_REVIEW.md).

# Kickoff prompt

Paste everything below the line into the agent, in a repo that already contains `CLAUDE.md`.

---

Read `CLAUDE.md` first. It contains the full spec, the pinned stack, the data contract, the repo-integration rules, and ten numbered failure modes that are rules rather than suggestions. Follow it exactly; if something in it seems wrong, say so and ask rather than deviating.

This is an existing repo with a teammate's detection-pipeline work already in it, or landing in it soon, under file names and a layout I don't know yet. **You will discover them, I will not describe them to you in advance.** I'm working on my own branch and must never touch their files or branch.

Build the fly-through viewer in eight phases, starting with a repo/branch setup phase. **Stop after each phase, report what you did, report the gate result, and wait for me before continuing.** Do not run the whole plan unattended.

I do not have real case data yet — the teammate is producing it. So Phase 1 generates synthetic data and every later phase is built against that. Real data is a file swap at the end.

## Phase 0 — Repo survey and branch setup

- Run `git status` and `git branch` to confirm current state. If not already on a dedicated branch for this work, create one off the current base (e.g. `git checkout -b flythrough-viewer`) and confirm the name with me before writing any files.
- Survey the repo: top-level tree, any existing README, and a search for likely detection-pipeline artifacts (`prediction.json`, `run.py`, files matching the challenge's `--image / --aorta-mask / --output` CLI, NIfTI files, existing `data/` directories, existing `requirements.txt`).
- Propose the viewer's namespaced top-level directory (default suggestion `flythrough/`) based on what would and wouldn't collide with what you found. Do not create any viewer files yet.

**Gate:** report the branch name, the repo survey findings (what exists, what's teammate-owned, what's empty/unclaimed), and the proposed directory name. Wait for me to confirm before Phase 1.

## Phase 1 — Synthetic data + verification harness

All paths below are relative to the confirmed viewer directory from Phase 0, not repo root.


- `tools/requirements.txt` with the exact dependency list from CLAUDE.md.
- `tools/frames.py` — rotation-minimizing frame via Rodrigues parallel transport. Signature `rmf(centerline, anterior) -> (tangents, up_vectors)`. Seed the up-vector by projecting `anterior` onto the plane perpendicular to the first tangent. Include a unit test asserting that over a helical centreline the up-vectors stay perpendicular to the tangent and never discontinuously flip (angle between consecutive up-vectors under 10 degrees).
- `tools/fake_case.py` — generates a plausible synthetic case into `data/subjectNNN/`:
  - a centreline: gentle S-curve, ~250 mm long, 1 mm spacing, coordinates offset into the hundreds so precision handling gets exercised
  - a diameter profile tapering 26 mm to 18 mm, with one localised narrowing
  - 4 branches at varied clock positions, radii 2–4 mm, confidences spread across 0.4–0.98
  - a watertight tube mesh from the centreline and diameter profile, plus modelled branch cylinders, per-vertex confidence colours baked in, exported as `mesh.glb`
  - a slice sprite sheet: synthetic cross-sections (bright disc on dark background, plus a small bright disc where a branch is present) in a grid, written as `slices.jpg`
  - `viewer.json` and a schema-valid `prediction.json`
- `tools/verify_assets.py` — every invariant and cross-check listed under "Verification" in CLAUDE.md. Non-zero exit on failure, one line per check.

Generate three synthetic cases and `data/index.json`.

**Gate:** `python tools/verify_assets.py data/subject001` exits 0 for all three cases. Paste the output. Also paste the frames unit test result.

## Phase 2 — Scene and mesh

`index.html` (layout containers, importmap pointing at `vendor/`), `src/scene.js`, `src/main.js`, `serve.sh`. Vendor three.js 0.160.0 and GLTFLoader into `vendor/` — download them now while network is available, since the final environment has none.

Load `mesh.glb`, apply `BackSide` + `vertexColors` to every material, parent a `PointLight` to the camera, add low ambient. Recentre geometry by subtracting `centerline[0]`. Place the camera at the first centreline point looking at the second.

**Gate:** `src/diagnostics.js` with `?selftest=1` implemented far enough to assert: glb loaded, mesh count > 0, every material is BackSide with vertexColors true, camera light is a child of the camera. Report `SELFTEST: n passed, m failed`. Then tell me to open it and confirm I can see a lit tube interior — I will report back.

## Phase 3 — Camera path and controls

`src/camera-path.js` — `CatmullRomCurve3` through the recentred centreline; `poseAt(t)` returning position, look-at target, and an up-vector interpolated (and renormalised) between the two nearest `up_vectors`. Never recompute a frame in JS.

`src/controls.js` — the full control table from CLAUDE.md. Held-keys `Set`, `dt`-integrated with `dt` clamped to 0.05, easing at 0.15, `t` clamped to `[0,1]`, `preventDefault` on `Arrow*` and `Space`. Left/Right jump to the previous/next branch by setting `tTarget = centerline_index / (N-1)`. Wire the range slider bidirectionally.

**Gate:** extend selftest to assert all six bindings registered, curve length within 10% of summed centreline segments, and camera positions at `t=0`, `t=0.5`, `t=1` all inside the lumen. Then ask me to confirm: does arrow-key travel feel smooth and does the view stay upright with no barrel-roll through the curves.

## Phase 4 — Diameter chart

`src/chart.js`, hand-written Canvas 2D. Polyline of `diameter_mm` across the full width, y-axis labelled in mm, vertical cursor at `t`, a tick per branch coloured by confidence, click-to-seek, and a live numeric readout of the current diameter. Redraw per frame; it is a few hundred points.

**Gate:** selftest asserts the canvas has non-blank pixel content and that a synthetic click at 50% width sets `tTarget` to 0.5 within tolerance.

## Phase 5 — CT slice panel

`src/slice.js` — compute the tile index from `t`, set `background-position` from `slice_grid.cols` and `tile_px`. No canvas. Add the `measured` badge.

**Gate:** selftest asserts the sprite image loaded with dimensions matching `cols * tile_px` by `ceil(N/cols) * tile_px`, and that `t=0` and `t=1` produce different, in-range background offsets.

## Phase 6 — HUD, branches, honesty layer

`src/clock.js` — SVG ring, 12 at the top, marking each branch at its `clock_hour` as it comes within range of `t`. `src/branches.js` — ring markers at each ostium oriented to `direction_xyz`, sprite labels with `depthTest: false`, and a side list showing instance id, radius, clock hour, length to bifurcation, and confidence; clicking a row jumps the camera.

Then the honesty layer: per-panel `measured`/`modelled` badges, the labelled confidence colour bar, and the verbatim legend line from CLAUDE.md. Distance-travelled readout in mm.

**Gate:** selftest asserts the legend string is present in the DOM verbatim, every panel has a badge, and marker count equals branch count.

## Phase 7 — Real data, README, fallback

When the teammate's real output is available, first check whether its shape matches the `viewer.json` / `prediction.json` contract in CLAUDE.md. If not, write `tools/adapt_teammate_output.py` to map their real file(s) into that contract — do not edit their files or their output format to fit you. `tools/prep_case.py` handles NIfTI to `mesh.glb` + `slices.jpg` + `viewer.json`, resampling to an identity-direction grid first. Write the viewer's own README (inside the viewer directory) with exactly two commands. Capture the demo fallback artifacts if a browser automation tool is already available — if not, tell me and I will record them.

**Gate:** all of "Definition of done" in CLAUDE.md, itemised with pass/fail. Also confirm: no file outside the viewer's own directory was modified (`git status` / `git diff --stat` against the base branch should show changes scoped to the viewer directory only, plus new files).

## Standing instructions

- You cannot see rendered WebGL output. Never tell me a visual feature works because the code looks right. Either verify it through `verify_assets.py` or the selftest harness, or explicitly ask me to look and say what I should be checking for.
- Before any write in every phase, if there's genuine ambiguity about whether a path belongs to the teammate, stop and ask rather than guessing.
- Never merge, rebase onto, or push to any branch other than the viewer's own without being explicitly asked.

Start with Phase 0.
