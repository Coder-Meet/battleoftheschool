import {
  createIcons,
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Box,
  Camera,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Crosshair,
  Database,
  Expand,
  Eye,
  FileJson,
  GitBranch,
  Layers,
  LoaderCircle,
  Map,
  PanelLeftClose,
  Play,
  RotateCcw,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TriangleAlert,
  X,
  ZoomIn,
  ZoomOut,
  Pause,
  Orbit,
} from "lucide";
import * as THREE from "three";
import { FEATURE_NAMES, ReviewStore } from "./reviews";
import type { ReviewLabel } from "./reviews";
import { AortaViewer } from "./viewer";
import { SliceViews } from "./slices";
import { selfCheckWarnings } from "./qc";
import type { Branch, Case, Point } from "./types";
import {
  branchName,
  COLORS,
  UNSCORED_COLOR,
  diameterGradientCSS,
  viridisGradientCSS,
} from "./types";
import "./style.css";

const icons = {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Box,
  Camera,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Crosshair,
  Database,
  Expand,
  Eye,
  FileJson,
  GitBranch,
  Layers,
  LoaderCircle,
  Map,
  PanelLeftClose,
  Play,
  RotateCcw,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TriangleAlert,
  X,
  ZoomIn,
  ZoomOut,
  Pause,
  Orbit,
};
const icon = (name: string) => `<i data-lucide="${name}"></i>`;
const refreshIcons = () =>
  createIcons({ icons, attrs: { "stroke-width": 1.65 } });
const $ = <T extends HTMLElement = HTMLElement>(selector: string) =>
  document.querySelector<T>(selector)!;
const escape = (value: string) =>
  value.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );

$("#app").innerHTML = `
  <nav class="rail" aria-label="Primary navigation">
    <a class="brand-mark" href="/" aria-label="Branchseed home">${icon("git-branch")}</a>
    <div class="rail-divider"></div>
    <button class="rail-button active" id="explore-nav" title="Aorta Explorer" aria-label="Aorta Explorer">${icon("box")}</button>
    <button class="rail-button" id="cases-nav" title="Toggle case library" aria-label="Toggle case library">${icon("database")}</button>
    <button class="rail-button" id="method-nav" title="Detection method" aria-label="Detection method">${icon("activity")}</button>
    <div class="rail-bottom">
      <button class="rail-button" id="help-nav" title="Help and controls" aria-label="Help and controls">${icon("circle-help")}</button>
      <div class="avatar" title="Local research workspace">TL</div>
    </div>
  </nav>
  <aside class="library">
    <div class="workspace-brand"><div><strong>branchseed<span>®</span></strong><p>VASCULAR INTELLIGENCE</p></div>
      <button class="icon-button" id="collapse-library" aria-label="Collapse case library">${icon("panel-left-close")}</button></div>
    <div class="workspace-label">WORKSPACE</div>
    <div class="project"><span class="project-icon">${icon("git-branch")}</span><div>Toralis Labs<small>Branch discovery challenge</small></div><span class="project-dot"></span></div>
    <div class="library-header"><span>Case library</span><span id="case-count" class="count">—</span></div>
    <label class="search-box">${icon("search")}<input id="case-search" placeholder="Search cases..." aria-label="Search cases" /><kbd>/</kbd></label>
    <div class="case-list" id="case-list"><div class="library-empty">Loading local cases…</div></div>
    <div class="library-foot"><div class="engine-icon">${icon("shield-check")}</div><div>On-device processing<small>CPU only · your data stays local</small></div><span class="status-dot"></span></div>
    <div class="version">BRANCHSEED <span>RESEARCH BUILD 1.0</span></div>
  </aside>
  <button id="library-backdrop" class="library-backdrop" aria-label="Close case library"></button>
  <main>
    <header class="topbar"><div class="breadcrumb">${icon("layers")}<span>Workspace</span>${icon("chevron-right")}<strong>Aorta Explorer</strong></div>
      <div class="topbar-right"><button class="research-pill" id="research-info"><span></span>Research prototype</button><div class="separator"></div><button id="help-top" class="icon-button" aria-label="Open help">${icon("circle-help")}</button></div></header>
    <section class="page-heading"><div><div class="eyebrow"><span class="live-dot"></span>ANATOMY WORKSPACE</div><h1>Aorta Explorer<span class="heading-dot">.</span></h1><p>See the anatomy. Trace every origin.</p></div>
      <div class="heading-actions"><button class="button secondary" id="method-button">${icon("activity")} View method</button><button class="button primary" id="export" disabled>${icon("arrow-down-to-line")} Export JSON</button></div></section>
    <section class="case-summary"><div class="case-identity"><span class="scan-icon">${icon("layers")}</span><div><h2 id="case-title">Select a case</h2><span id="case-subtitle">CT angiography · parent aorta</span></div></div>
      <div class="summary-divider"></div><div class="metric"><span>Detected branches</span><strong id="metric-branches">—</strong></div>
      <div class="metric"><span>Aortic coverage</span><strong id="metric-coverage">—</strong></div>
      <div class="metric"><span>Native voxel size</span><strong id="metric-spacing">—</strong></div>
      <div class="metric"><span>Detection time</span><strong id="metric-runtime">—</strong></div>
      <div class="case-arrows"><button class="icon-button" id="previous-case" aria-label="Previous case">${icon("arrow-left")}</button><button class="icon-button" id="next-case" aria-label="Next case">${icon("arrow-right")}</button></div>
    </section>
    <section class="analysis-layout">
      <div class="visual-column">
        <div class="visual-panel">
          <div class="view-toolbar"><div class="view-tabs" role="tablist" aria-label="Visualization mode">
            <button class="view-tab active" data-mode="3d" role="tab" aria-selected="true">${icon("box")}<span>3D reconstruction</span></button>
            <button class="view-tab" data-mode="ct" role="tab" aria-selected="false">${icon("layers")}<span>CT evidence</span></button>
            <button class="view-tab" data-mode="map" role="tab" aria-selected="false">${icon("map")}<span>Wall map</span></button>
          </div><span class="coordinate-label">LPS <span>·</span> mm</span><button class="icon-button" id="fullscreen" title="Expand workspace" aria-label="Expand workspace">${icon("expand")}</button></div>
          <div class="model-area" id="model-area">
            <div id="viewer"></div>
            <div class="model-caption"><span class="small-label">PARENT AORTA + DAUGHTER INSTANCES</span><div><span class="live-dot"></span>CT-derived surface</div></div>
            <div class="model-tools"><button class="tool-button" id="reset-camera" title="Reset camera" aria-label="Reset camera">${icon("rotate-ccw")}</button><button class="tool-button" id="zoom-in" title="Zoom in" aria-label="Zoom in">${icon("zoom-in")}</button><button class="tool-button" id="zoom-out" title="Zoom out" aria-label="Zoom out">${icon("zoom-out")}</button><div></div><button class="tool-button" id="rotate" title="Auto rotate" aria-label="Auto rotate">${icon("orbit")}</button><div></div><button class="tool-button" id="export-visual-check" title="Export visual check (aorta, ostia, direction arrows)" aria-label="Export visual check image">${icon("camera")}</button></div>
            <div class="orientation"><canvas width="90" height="90" aria-label="Camera-linked anatomical orientation"></canvas><small>LPS · camera-linked axes</small></div>
            <div class="legend-stack">
              <div class="color-legend" id="diameter-legend" aria-label="Vessel diameter color scale"><span class="small-label">VESSEL DIAMETER</span><div class="legend-bar" id="diameter-legend-bar"></div><div class="legend-current" id="diameter-current-row" hidden><span>At this position</span><strong id="diameter-current">— mm</strong></div><div class="legend-values"><span id="diameter-min">— mm</span><span id="diameter-max">— mm</span></div><div class="legend-ticks"><span>Thinnest</span><span>Widest</span></div></div>
              <div class="color-legend" id="evidence-legend" aria-label="Detection evidence color scale"><span class="small-label">DETECTION EVIDENCE</span><div class="legend-bar" id="legend-bar"></div><div class="legend-ticks"><span>0.0 low</span><span>1.0 high</span></div><div class="legend-unscored"><span class="legend-swatch" id="legend-swatch"></span>Unscored</div></div>
            </div>
            <div class="model-bottom"><span>${icon("orbit")} Drag to orbit <b>·</b> Scroll to zoom</span><button class="fly-button" id="flythrough">${icon("play")} Enter aorta <span>3D TOUR</span></button></div>
            <div class="flight-controls" id="flight-controls" hidden><button class="icon-button" id="play-flight" aria-label="Play or pause fly-through">${icon("play")}</button><span>Endoluminal view</span><input type="range" id="flight-position" min="2" max="98" value="10" aria-label="Position inside aorta" /><select id="flight-speed" aria-label="Fly-through speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><button class="icon-button" id="exit-flight" aria-label="Exit fly-through">${icon("x")}</button></div>
          </div>
          <div class="wall-map" id="wall-map" hidden></div>
          <div class="ct-mode-heading" id="ct-mode-heading" hidden><strong>Linked multiplanar review</strong><span>Scroll through slices or click to move the crosshair. Select a branch to locate its ostium.</span></div>
          <div class="layers-bar"><span>${icon("layers")} Layers</span><label><input id="parent-layer" type="checkbox" checked /><span class="swatch aorta"></span>Parent aorta</label><label><input id="labels-layer" type="checkbox" checked /><span class="swatch daughters"></span>Branch labels</label><label><input id="centerline-layer" type="checkbox" /><span class="swatch line"></span>Centerline</label><label><input id="diameter-layer" type="checkbox" /><span class="swatch diameter"></span>Diameter color</label><div class="opacity"><span>Opacity</span><input id="opacity" type="range" min="15" max="100" value="92" aria-label="Aorta opacity" /></div></div>
        </div>
        <div class="evidence-panel" id="evidence-panel"><div class="section-heading"><h3>${icon("crosshair")} CT evidence <span>LINKED TO SELECTION</span></h3><label>Window <select id="ct-window" aria-label="CT window"><option value="cta">Angiography</option><option value="soft">Soft tissue</option><option value="bone">Bone</option></select></label></div><div class="slices" id="slices"></div></div>
      </div>
      <aside class="inspector"><div class="inspector-heading"><div><h3>Branch instances <span id="branch-count" class="count">0</span></h3><p>Direct daughters of the parent aorta</p></div>${icon("git-branch")}</div>
        <div class="review-toolbar"><label>Show <select id="branch-filter" aria-label="Filter branch reviews"><option value="all">All candidates</option><option value="unreviewed">Needs review</option><option value="confirmed">Confirmed</option><option value="rejected">Rejected</option></select></label><button id="export-reviews" class="icon-button" title="Export training reviews for all cases" aria-label="Export training reviews">${icon("arrow-down-to-line")}</button><span id="review-progress" role="status"></span><span id="qc-summary" class="qc-summary" role="status" hidden></span></div>
        <div class="branch-list" id="branch-list"><div class="empty-branches">Analyze a case to discover branches.</div></div>
        <div class="branch-details" id="branch-details"><div class="no-selection">${icon("crosshair")}<h4>Follow an origin</h4><p>Select a branch to inspect its coordinates and proximal path.</p></div></div>
        <div class="inspector-note">${icon("circle-help")}<p>Detections are candidates for review.<br>Evidence scores are not probabilities.</p></div>
      </aside>
      <div class="loading-overlay" id="loading"><div class="loading-content"><div class="loading-orb">${icon("git-branch")}</div><h3 id="loading-title">Opening your workspace</h3><p id="loading-description">Connecting to the local processing engine.</p><div class="loading-stages"><span>CT + mask</span><i></i><span>Vessel evidence</span><i></i><span>Branch paths</span></div><button class="button secondary" id="retry" hidden>Try again</button></div></div>
    </section>
    <footer class="page-footer"><span><span class="status-dot"></span>Local CPU engine <b>·</b> No cloud inference</span><span>Physical coordinates preserved <b>·</b> Toralis Labs Challenge</span></footer>
  </main>
  <dialog id="help-dialog"><div class="dialog-heading"><span>${icon("git-branch")} THE BRANCHSEED APPROACH</span><button class="icon-button" id="close-help" aria-label="Close help">${icon("x")}</button></div>
    <h2>Anatomy first.<br><span>Evidence at every origin.</span></h2><p class="dialog-intro">A CPU-only research pipeline that uses the supplied parent aorta as an anchor to discover the arteries that leave it.</p>
    <div class="method-steps"><div><b>01</b><section><h4>Understand the scan</h4><p>Validate physical geometry, crop around the parent and estimate the patient’s contrast-filled blood intensity.</p></section></div><div><b>02</b><section><h4>Discover wall connections</h4><p>Combine adaptive intensity with multiscale Sato vessel enhancement. Separate candidate openings and suppress cropped aortic ends.</p></section></div><div><b>03</b><section><h4>Trace and measure</h4><p>Follow supported proximal paths. Place the seed 5 mm along the path and measure radius and outward direction in physical space.</p></section></div></div>
    <div id="case-diagnostics"></div>
    <div class="research-note"><strong>Research prototype · not validated for clinical use</strong><p>No daughter reference annotations are included in the dataset. Precision and recall are unmeasured. Low-contrast vessels, shared trunks, adjacent veins, and short branches need particular review. Colored tubes show estimated paths, not full daughter segmentations. The interior tour follows an approximate parent curve.</p></div>
    <div class="help-shortcuts"><span><kbd>/</kbd> Search cases</span><span><kbd>R</kbd> Reset camera</span><span><kbd>Esc</kbd> Close dialog / exit tour</span></div>
  </dialog>
  <div class="toast" id="toast" role="status" hidden></div>
`;
refreshIcons();
$("#legend-bar").style.background = viridisGradientCSS();
$("#legend-swatch").style.background = UNSCORED_COLOR;
$("#diameter-legend-bar").style.background = diameterGradientCSS();

let data: Case | undefined;
let cases: { id: string; available: boolean }[] = [];
let selectedCase = "";
let selectedBranch = "";
let loadSequence = 0;
let loadController: AbortController | undefined;
let mode = "3d";
let flythrough = false;
let viewer: AortaViewer | undefined;
const reviews = new ReviewStore();
const slices = new SliceViews($("#slices"));
try {
  viewer = new AortaViewer(
    $("#viewer"),
    selectBranch,
    $(".orientation canvas"),
  );
  viewer.onFlightProgress = (progress) => {
    $<HTMLInputElement>("#flight-position").value = String(progress * 100);
  };
  viewer.onFlightPlaying = (playing) => {
    $("#play-flight").innerHTML = icon(playing ? "pause" : "play");
    refreshIcons();
  };
  viewer.onWallDiameterRange = (minMm, maxMm) => {
    $("#diameter-min").textContent = `${minMm.toFixed(1)} mm`;
    $("#diameter-max").textContent = `${maxMm.toFixed(1)} mm`;
  };
  viewer.onCurrentDiameter = (mm) => {
    $("#diameter-current").textContent = `${mm.toFixed(1)} mm`;
  };
} catch {
  $("#viewer").innerHTML =
    '<div class="webgl-error">3D rendering is unavailable in this browser.<br>CT evidence, the wall map, and JSON export are still available.</div>';
}

async function request(path: string, method = "GET", signal?: AbortSignal) {
  return fetch(path, {
    method,
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(30000)])
      : AbortSignal.timeout(30000),
  });
}

async function api<T>(
  path: string,
  method = "GET",
  signal?: AbortSignal,
): Promise<T> {
  const response = await request(path, method, signal);
  if (!response.ok) {
    const error = (await response
      .json()
      .catch(() => ({ error: `Request failed (${response.status})` }))) as {
      error?: string;
    };
    throw new Error(error.error || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function displayCase(id: string) {
  return /^subject\d+$/.test(id) ? `Subject ${id.replace("subject", "")}` : id;
}

function renderCases() {
  const filter = $<HTMLInputElement>("#case-search")
    .value.toLowerCase()
    .replace(/\s/g, "");
  const visible = cases.filter((c) => c.id.toLowerCase().includes(filter));
  $("#case-list").innerHTML =
    visible
      .map(
        (c) => `
    <button class="case-item ${c.id === selectedCase ? "selected" : ""}" data-case="${escape(c.id)}" ${c.available ? "" : "disabled"} title="${c.available ? `Open ${escape(displayCase(c.id))}` : "Run git lfs pull to download this case"}">
      <span class="case-thumbnail">${icon("layers")}</span><div><strong>${escape(displayCase(c.id))}</strong><small>${c.available ? "CT angiography" : "LFS download needed"}</small></div><span class="case-state">${c.id === selectedCase ? icon("chevron-right") : "<span></span>"}</span>
    </button>
  `,
      )
      .join("") || '<div class="library-empty">No matching cases</div>';
  $("#case-count").textContent = String(cases.length).padStart(2, "0");
  $("#case-list")
    .querySelectorAll<HTMLButtonElement>("[data-case]")
    .forEach((button) => {
      button.onclick = () => void loadCase(button.dataset.case!);
    });
  refreshIcons();
}

function setLoading(title: string, description: string, error = false) {
  $("#loading").hidden = false;
  $("#loading").classList.toggle("error", error);
  $("#loading-title").textContent = title;
  $("#loading-description").textContent = description;
  $("#retry").hidden = !error;
  $<HTMLButtonElement>("#export").disabled = true;
  $("#case-title").textContent = displayCase(selectedCase);
}

async function loadCase(id: string) {
  const sequence = ++loadSequence;
  loadController?.abort();
  loadController = new AbortController();
  const signal = loadController.signal;
  const deadline = Date.now() + 180000;
  selectedCase = id;
  data = undefined;
  $("#case-subtitle").textContent = "Loading CT and parent mask…";
  renderCases();
  if (window.matchMedia("(max-width: 1080px)").matches)
    document.body.classList.remove("library-collapsed");
  if (flythrough) setFlythrough(false);
  setLoading(
    `Analyzing ${displayCase(id)}`,
    "Reading the CT and parent mask. Processing happens on this machine.",
  );
  for (const metric of ["branches", "coverage", "spacing", "runtime"])
    $(`#metric-${metric}`).textContent = "—";
  try {
    let status = await api<{ status: string; error?: string }>(
      `/api/cases/${id}/analyze`,
      "POST",
      signal,
    );
    while (status.status !== "ready") {
      if (sequence !== loadSequence) return;
      if (Date.now() > deadline)
        throw new Error("Analysis timed out. Check the CPU engine and retry.");
      if (status.status === "failed")
        throw new Error(status.error || "Analysis failed.");
      if (status.status === "idle")
        throw new Error("This case left the cache. Try analyzing it again.");
      $("#loading-description").textContent =
        status.status === "queued"
          ? "Waiting for the CPU engine. One case is processed at a time."
          : "Enhancing vessels, tracing wall connections, and building the 3D surface.";
      await new Promise((resolve) => setTimeout(resolve, 700));
      status = await api(`/api/cases/${id}/status`, "GET", signal);
    }
    const [metadata, ctResponse, maskResponse] = await Promise.all([
      api<Case>(`/api/cases/${id}`, "GET", signal),
      request(`/api/cases/${id}/ct`, "GET", signal),
      request(`/api/cases/${id}/mask`, "GET", signal),
    ]);
    if (!ctResponse.ok || !maskResponse.ok)
      throw new Error("Volume data is no longer cached. Try again.");
    const [ct, mask] = await Promise.all([
      ctResponse.arrayBuffer(),
      maskResponse.arrayBuffer(),
    ]);
    if (sequence !== loadSequence) return;
    data = metadata;
    if (JSON.stringify(data.feature_names) !== JSON.stringify(FEATURE_NAMES))
      toast(
        "Server feature contract differs from this page. Rebuild web/ before exporting reviews.",
      );
    $(".eyebrow").innerHTML =
      data.profile === "review"
        ? '<span class="live-dot"></span>REVIEW POOL · LOOSE DETECTOR · EVERY CANDIDATE NEEDS A VERDICT'
        : '<span class="live-dot"></span>ANATOMY WORKSPACE';
    const removed = reviews.reconcile(id, data.branches);
    if (removed)
      toast(
        `${removed} outdated review(s) removed. Please review the updated candidates.`,
      );
    viewer?.load(data);
    slices.load(data, ct, mask);
    history.replaceState(null, "", `#case=${encodeURIComponent(id)}`);
    $("#metric-branches").innerHTML =
      `${String(data.branches.length).padStart(2, "0")}<span class="metric-tag">instances</span>`;
    $("#metric-coverage").innerHTML =
      `${data.coverage_mm.toFixed(1)}<small> mm</small>`;
    $("#metric-spacing").innerHTML =
      `${data.native_spacing_xyz.map((n) => n.toFixed(2)).join(" × ")}<small> mm</small>`;
    $("#metric-runtime").innerHTML =
      `${data.diagnostics.timings.total_s.toFixed(1)}<small> s</small><span class="cpu-tag">CPU</span>`;
    $("#case-subtitle").textContent =
      `${data.native_size_xyz.join(" × ")} voxels · NIfTI`;
    $("#loading").hidden = true;
    $<HTMLButtonElement>("#export").disabled = false;
    selectedBranch = data.branches[0]?.instance_id || "";
    renderBranches();
    renderMap();
    if (selectedBranch) selectBranch(selectedBranch);
    else
      $("#branch-details").innerHTML =
        '<div class="no-selection"><h4>No eligible branches detected</h4><p>Inspect the CT evidence. An empty prediction does not establish that no branches exist.</p></div>';
    $("#case-diagnostics").innerHTML =
      `<div class="diagnostic-grid"><div><span>Wall candidates</span><strong>${data.diagnostics.candidates}</strong></div><div><span>Parent blood</span><strong>${Math.round(data.diagnostics.blood_model.median_hu)} HU</strong></div><div><span>Vessel enhancement</span><strong>${data.diagnostics.timings.enhance_s}s</strong></div></div>`;
    applyLayers();
    setMode(mode);
  } catch (error) {
    if (sequence !== loadSequence) return;
    setLoading(
      "Couldn’t analyze this case",
      error instanceof Error ? error.message : "Unexpected error.",
      true,
    );
  }
}

function renderBranches() {
  if (!data) return;
  const filter = $<HTMLSelectElement>("#branch-filter").value;
  const pending = data.branches.filter(
    (b) => reviews.status(data!.case_id, b) === "unreviewed",
  );
  $("#review-progress").textContent =
    `${data.branches.length - pending.length}/${data.branches.length} reviewed`;
  $("#branch-count").textContent = String(data.branches.length);
  const flaggedCount = data.branches.filter((b) => selfCheckWarnings(b).length).length;
  $("#qc-summary").hidden = flaggedCount === 0;
  $("#qc-summary").innerHTML = `${icon("triangle-alert")} ${flaggedCount} flagged`;
  $("#branch-list").innerHTML =
    data.branches
      .map((branch, i) => {
        const status = reviews.status(data!.case_id, branch);
        if (filter !== "all" && status !== filter) return "";
        const flags = selfCheckWarnings(branch);
        return `
    <button class="branch-item ${branch.instance_id === selectedBranch ? "selected" : ""}" data-branch="${branch.instance_id}" style="--branch-color:${COLORS[i % COLORS.length]}">
      <span class="branch-symbol">${icon(status === "confirmed" ? "check" : status === "rejected" ? "x" : "git-branch")}</span><div><strong>${branchName(branch.instance_id)}</strong><small>${status === "unreviewed" ? `Radius ${branch.radius_mm.toFixed(2)} mm` : status}</small></div>${flags.length ? `<span class="branch-qc-flag" title="${escape(flags.join(" "))}">${icon("triangle-alert")}</span>` : ""}<span class="branch-evidence">${branch.evidence_score.toFixed(2)}<small>evidence</small></span>
    </button>
  `;
      })
      .join("") ||
    `<div class="empty-branches">${data.branches.length ? "No candidates match this filter." : "No eligible branches detected.<br>Review the linked CT slices."}</div>`;
  $("#branch-list")
    .querySelectorAll<HTMLButtonElement>("[data-branch]")
    .forEach(
      (button) => (button.onclick = () => selectBranch(button.dataset.branch!)),
    );
  refreshIcons();
}

function selectBranch(id: string) {
  const branch = data?.branches.find((b) => b.instance_id === id);
  if (!branch || !data) return;
  selectedBranch = id;
  viewer?.select(id);
  slices.focus(branch.ostium_xyz_mm);
  renderBranches();
  const color = COLORS[data.branches.indexOf(branch) % COLORS.length];
  const coords = (point: Point) =>
    point
      .map(
        (n, i) =>
          `<div><span>${["X", "Y", "Z"][i]}</span>${n.toFixed(2)}</div>`,
      )
      .join("");
  const fmt = (n: number | undefined) =>
    typeof n === "number" ? n.toFixed(2) : "—";
  $("#branch-details").innerHTML = `
    <div class="detail-heading"><span class="small-label">SELECTED INSTANCE</span><button class="icon-button" id="focus-branch" title="Focus in 3D" aria-label="Focus selected branch in 3D">${icon("crosshair")}</button></div>
    <h3><span class="color-dot" style="background:${color}"></span>${branchName(id)}<span class="parent-link">aorta</span></h3>
    <div class="detail-metrics"><div><span>Lumen radius</span><strong>${branch.radius_mm.toFixed(2)}<small> mm</small></strong></div><div><span>Seed offset</span><strong>5.00<small> mm</small></strong></div></div>
    <div class="coordinate-heading">Ostium coordinates <span>LPS · mm</span></div><div class="coordinates">${coords(branch.ostium_xyz_mm)}</div>
    <div class="coordinate-heading">Daughter seed <span>LPS · mm</span></div><div class="coordinates">${coords(branch.seed_xyz_mm)}</div>
    <div class="direction-row"><span>${icon("git-branch")} Outward direction</span><code>[${branch.direction_xyz.map((n) => n.toFixed(2)).join(", ")}]</code></div>
    <div class="detail-metrics"><div><span>Path HU vs blood</span><strong>${fmt(branch.features.path_hu_relative)}</strong></div><div><span>Bone distance</span><strong>${fmt(branch.features.bone_distance_mm)}<small> mm</small></strong></div></div>
    <div class="detail-metrics"><div><span>Angle to aorta</span><strong>${fmt(branch.features.parent_angle_degrees)}<small> °</small></strong></div><div><span>Wall gap</span><strong>${fmt(branch.features.connector_gap)}</strong></div></div>
    <div class="evidence-score"><span>Geometric evidence <strong>${branch.evidence_score.toFixed(2)}</strong></span><div><i style="width:${branch.evidence_score * 100}%;background:${color}"></i></div><small>Heuristic score · uncalibrated</small></div>
    <button class="button inspect-button" id="inspect-ct">${icon("layers")} Inspect CT evidence ${icon("arrow-right")}</button>
    ${branch.warnings.length ? `<p class="branch-warning">${branch.warnings.map(escape).join("<br>")}</p>` : ""}
    ${selfCheckWarnings(branch).length ? `<p class="branch-qc">${icon("triangle-alert")}${selfCheckWarnings(branch).map(escape).join("<br>")}</p>` : ""}
    <section class="review-actions" aria-label="Candidate review"><span class="small-label">HUMAN REVIEW · ${reviews.status(data.case_id, branch).toUpperCase()}</span><div>
      <button class="button secondary" data-review="confirmed" aria-pressed="${reviews.status(data.case_id, branch) === "confirmed"}">Confirm</button>
      <button class="button secondary" data-review="rejected" aria-pressed="${reviews.status(data.case_id, branch) === "rejected"}">Reject</button>
      <button class="button secondary" data-review="unreviewed">Clear</button></div>
      <button class="button inspect-button" id="next-review">Next unreviewed ${icon("arrow-right")}</button>
      <small>Saved in this browser. Export reviews to train a candidate classifier. Raw prediction export stays unchanged.</small>
    </section>
  `;
  $("#branch-details")
    .querySelectorAll<HTMLButtonElement>("[data-review]")
    .forEach((button) => {
      button.onclick = () => {
        const persisted = reviews.set(
          data!.case_id,
          branch,
          button.dataset.review as ReviewLabel | "unreviewed",
        );
        selectBranch(id);
        if (!persisted)
          toast(
            "Browser storage unavailable. Export reviews before closing this page.",
          );
      };
    });
  $("#next-review").onclick = () => {
    if (!data) return;
    const index = data.branches.indexOf(branch);
    const ordered = [
      ...data.branches.slice(index + 1),
      ...data.branches.slice(0, index + 1),
    ];
    const next = ordered.find(
      (b) => reviews.status(data!.case_id, b) === "unreviewed",
    );
    if (next) selectBranch(next.instance_id);
    else toast("All candidates in this case have been reviewed.");
  };
  $("#focus-branch").onclick = () => {
    if (flythrough) setFlythrough(false);
    setMode("3d");
    viewer?.focus(id);
  };
  $("#inspect-ct").onclick = () => setMode("ct");
  $("#wall-map")
    .querySelectorAll("[data-map-branch]")
    .forEach((element) => {
      element.classList.toggle(
        "selected",
        element.getAttribute("data-map-branch") === id,
      );
    });
  refreshIcons();
}

function wallPosition(branch: Branch, centerline: Point[]) {
  const point = new THREE.Vector3(...branch.ostium_xyz_mm);
  const centers = centerline.map((p) => new THREE.Vector3(...p));
  let closest = 0;
  centers.forEach((center, i) => {
    if (center.distanceTo(point) < centers[closest].distanceTo(point))
      closest = i;
  });
  const tangent = centers[Math.min(closest + 2, centers.length - 1)]
    .clone()
    .sub(centers[Math.max(0, closest - 2)])
    .normalize();
  const reference =
    Math.abs(tangent.dot(new THREE.Vector3(1, 0, 0))) > 0.9
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(1, 0, 0);
  const horizontal = reference
    .clone()
    .addScaledVector(tangent, -reference.dot(tangent))
    .normalize();
  const vertical = tangent.clone().cross(horizontal).normalize();
  const offset = point.clone().sub(centers[closest]);
  const angle = Math.atan2(offset.dot(vertical), offset.dot(horizontal));
  let distance = 0,
    total = 0;
  for (let i = 1; i < centers.length; i++) {
    const step = centers[i].distanceTo(centers[i - 1]);
    total += step;
    if (i <= closest) distance += step;
  }
  return { angle, distance, total };
}

function renderMap() {
  if (!data) return;
  const points = data.branches
    .map((b, i) => {
      const p = wallPosition(b, data!.centerline);
      const x = 85 + ((p.angle + Math.PI) / (Math.PI * 2)) * 500;
      const y = 45 + (1 - p.distance / Math.max(p.total, 1)) * 255;
      return `<g data-map-branch="${b.instance_id}" class="map-point" tabindex="0" role="button" aria-label="Inspect ${branchName(b.instance_id)}"><circle cx="${x}" cy="${y}" r="${Math.max(5, b.radius_mm * 3)}" fill="${COLORS[i % COLORS.length]}" fill-opacity=".22" stroke="${COLORS[i % COLORS.length]}"/><circle cx="${x}" cy="${y}" r="2.5" fill="${COLORS[i % COLORS.length]}"/><text x="${x + 14}" y="${y + 4}">${b.instance_id.replace("branch_", "")}</text></g>`;
    })
    .join("");
  $("#wall-map").innerHTML =
    `<div class="map-heading"><div><span class="small-label">UNWRAPPED AORTIC WALL</span><h3>Every origin, in one view.</h3></div><span class="map-key"><i></i>Radius-scaled origins</span></div><svg viewBox="0 0 665 355" aria-label="Aortic wall branch origin map">
    <defs><pattern id="map-grid" width="62.5" height="51" patternUnits="userSpaceOnUse" x="85" y="45"><path d="M 62.5 0 L 0 0 0 51" fill="none" stroke="#24303b" stroke-width="1" /></pattern></defs>
    <rect x="85" y="45" width="500" height="255" fill="url(#map-grid)" stroke="#303b45"/><rect x="85" y="45" width="500" height="14" fill="#a88742" opacity=".08"/><rect x="85" y="286" width="500" height="14" fill="#a88742" opacity=".08"/>
    <text x="335" y="340" text-anchor="middle">Circumferential angle</text><text transform="translate(26,175) rotate(-90)" text-anchor="middle">Parent arc length (mm)</text>
    ${[-180, -90, 0, 90, 180].map((a, i) => `<text x="${85 + i * 125}" y="319" text-anchor="middle">${a}°</text>`).join("")}
    ${[0, 0.5, 1].map((f) => `<text x="70" y="${304 - f * 255}" text-anchor="end">${Math.round(data!.coverage_mm * f)}</text>`).join("")}
    ${points}</svg><p class="map-footnote">Approximate parent coordinate frame · select an origin to synchronize CT evidence.</p>`;
  $("#wall-map")
    .querySelectorAll<SVGGElement>("[data-map-branch]")
    .forEach((point) => {
      const activate = () => selectBranch(point.dataset.mapBranch!);
      point.onclick = activate;
      point.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      };
    });
}

function setMode(next: string) {
  mode = next;
  if (flythrough && next !== "3d") setFlythrough(false);
  $("#model-area").hidden = next !== "3d";
  $("#wall-map").hidden = next !== "map";
  $("#ct-mode-heading").hidden = next !== "ct";
  $(".visual-column").classList.toggle("ct-mode", next === "ct");
  document
    .querySelectorAll<HTMLButtonElement>("[data-mode]")
    .forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === next);
      button.setAttribute(
        "aria-selected",
        String(button.dataset.mode === next),
      );
    });
  requestAnimationFrame(() => slices.draw());
}

function setFlythrough(enabled: boolean) {
  if (!data && enabled) return;
  flythrough = enabled;
  if (enabled) setMode("3d");
  viewer?.setFlythrough(enabled);
  if (!enabled) applyLayers();
  $("#flight-controls").hidden = !enabled;
  $("#diameter-current-row").hidden = !enabled;
  $(".model-bottom").classList.toggle("touring", enabled);
  $(".model-caption").classList.toggle("touring", enabled);
  $(".orientation").classList.toggle("touring", enabled);
  $("#play-flight").innerHTML = icon("play");
  refreshIcons();
}

function applyLayers() {
  viewer?.setParent($<HTMLInputElement>("#parent-layer").checked);
  viewer?.setLabels($<HTMLInputElement>("#labels-layer").checked);
  viewer?.setCenterline($<HTMLInputElement>("#centerline-layer").checked);
  viewer?.setOpacity(Number($<HTMLInputElement>("#opacity").value) / 100);
  slices.setOverlay($<HTMLInputElement>("#parent-layer").checked);
  const diameterOn = $<HTMLInputElement>("#diameter-layer").checked;
  viewer?.setDiameterColoring(diameterOn);
  $("#diameter-legend").hidden = !diameterOn;
}

function showHelp() {
  $<HTMLDialogElement>("#help-dialog").showModal();
}
let toastTimer: ReturnType<typeof setTimeout>;
function toast(message: string) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => {
    $("#toast").hidden = true;
  }, 3000);
}
function download() {
  if (!data) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data.prediction, null, 2)], {
      type: "application/json",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${data.case_id}_prediction.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Prediction exported in physical LPS coordinates.");
}
function exportVisualCheck() {
  if (!data || !viewer) return;
  if (flythrough) setFlythrough(false);
  setMode("3d");
  const anchor = document.createElement("a");
  anchor.href = viewer.captureImage();
  anchor.download = `${data.case_id}_visual_check.png`;
  anchor.click();
  toast("Visual check exported: aorta, ostia and direction arrows.");
}
function stepCase(direction: number) {
  const available = cases.filter((c) => c.available);
  const index = available.findIndex((c) => c.id === selectedCase);
  const next =
    available[(index + direction + available.length) % available.length];
  if (next) void loadCase(next.id);
}

$("#case-search").oninput = renderCases;
$("#branch-filter").onchange = renderBranches;
$("#export-reviews").onclick = () => {
  const payload = reviews.export();
  if (!payload.records.length)
    return toast("Confirm or reject candidates before exporting reviews.");
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "branchseed-reviews.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast(
    `Exported ${payload.records.length} candidate reviews across all reviewed cases.`,
  );
};
$("#export").onclick = download;
$("#retry").onclick = () => {
  if (selectedCase) void loadCase(selectedCase);
  else void initialize();
};
$("#previous-case").onclick = () => stepCase(-1);
$("#next-case").onclick = () => stepCase(1);
$("#collapse-library").onclick = $("#cases-nav").onclick = () =>
  document.body.classList.toggle("library-collapsed");
$("#library-backdrop").onclick = () =>
  document.body.classList.remove("library-collapsed");
$("#explore-nav").onclick = () => setMode("3d");
for (const id of [
  "method-nav",
  "help-nav",
  "method-button",
  "help-top",
  "research-info",
])
  $(`#${id}`).onclick = showHelp;
$("#close-help").onclick = () => $<HTMLDialogElement>("#help-dialog").close();
$("#help-dialog").onclick = (event) => {
  if (event.target === $("#help-dialog"))
    $<HTMLDialogElement>("#help-dialog").close();
};
document
  .querySelectorAll<HTMLButtonElement>("[data-mode]")
  .forEach((button) => (button.onclick = () => setMode(button.dataset.mode!)));
$("#reset-camera").onclick = () => {
  if (flythrough) setFlythrough(false);
  viewer?.reset();
};
$("#zoom-in").onclick = () => viewer?.zoom(0.8);
$("#zoom-out").onclick = () => viewer?.zoom(1.25);
$("#rotate").onclick = () =>
  $("#rotate").classList.toggle("active", viewer?.toggleOrbit());
$("#export-visual-check").onclick = exportVisualCheck;
$("#fullscreen").onclick = () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else
    void $(".visual-column")
      .requestFullscreen()
      .catch(() => toast("Fullscreen is not available in this preview."));
};
$("#flythrough").onclick = () => setFlythrough(true);
$("#exit-flight").onclick = () => setFlythrough(false);
$("#play-flight").onclick = () => {
  const playing = viewer?.playFlight();
  $("#play-flight").innerHTML = icon(playing ? "pause" : "play");
  refreshIcons();
};
$<HTMLInputElement>("#flight-position").oninput = (event) =>
  viewer?.setFlightPosition(
    Number((event.target as HTMLInputElement).value) / 100,
  );
$<HTMLSelectElement>("#flight-speed").onchange = (event) =>
  viewer?.setSpeed(Number((event.target as HTMLSelectElement).value));
for (const id of [
  "parent-layer",
  "labels-layer",
  "centerline-layer",
  "diameter-layer",
  "opacity",
])
  $(`#${id}`).oninput = applyLayers;
$<HTMLSelectElement>("#ct-window").onchange = (event) =>
  slices.setWindow((event.target as HTMLSelectElement).value);
window.addEventListener("keydown", (event) => {
  if (
    event.ctrlKey ||
    event.metaKey ||
    event.altKey ||
    event.target instanceof HTMLInputElement ||
    event.target instanceof HTMLSelectElement
  )
    return;
  if (event.key === "/") {
    event.preventDefault();
    document.body.classList.toggle(
      "library-collapsed",
      window.matchMedia("(max-width: 1080px)").matches,
    );
    $("#case-search").focus();
  }
  if (event.key.toLowerCase() === "r") {
    if (flythrough) setFlythrough(false);
    viewer?.reset();
  }
  if (event.key === "Escape" && flythrough) setFlythrough(false);
  if (
    event.key === "ArrowUp" ||
    event.key === "ArrowDown" ||
    event.key === "ArrowLeft" ||
    event.key === "ArrowRight"
  ) {
    event.preventDefault();
    viewer?.handleKeyDown(event.key);
  }
});
window.addEventListener("keyup", (event) => {
  if (event.key === "ArrowUp" || event.key === "ArrowDown")
    viewer?.handleKeyUp(event.key);
});

async function initialize() {
  try {
    cases = await api("/api/cases");
    renderCases();
    const requested = new URLSearchParams(location.hash.slice(1)).get("case");
    const first =
      cases.find((c) => c.available && c.id === requested) ||
      cases.find((c) => c.available);
    if (first) await loadCase(first.id);
    else
      setLoading(
        "No CT cases available",
        "Add paired orig*.nii and mask*.nii files under the data directory, or run git lfs pull for the supplied dataset.",
        true,
      );
  } catch (error) {
    setLoading(
      "Local engine unavailable",
      error instanceof Error
        ? error.message
        : "Start the Explorer server and retry.",
      true,
    );
  }
}
void initialize();
