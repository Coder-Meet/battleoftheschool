import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Branch, Case, Point } from "./types";
import { branchName, COLORS, diameterColorRGB, evidenceColor } from "./types";
import {
  ARROW_SPEED,
  MAX_PROGRESS,
  MIN_PROGRESS,
  clamp,
  closestPolylinePosition,
  nextBranchProgress,
  percentileRangeExcluding,
  progressForPoint,
  ringIndexAt,
  ringRadii,
  ringsNear,
  seedFlightUp,
  transportFlightUp,
} from "./flight-math";

const viewPoint = (p: Point) => new THREE.Vector3(p[0], p[2], -p[1]);

export class AortaViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(35, 1, 0.1, 3000);
  private controls: OrbitControls;
  private anatomy = new THREE.Group();
  private branchObjects = new THREE.Group();
  private parent?: THREE.Mesh<THREE.BufferGeometry, THREE.Material>;
  private tissueMaterial?: THREE.MeshPhysicalMaterial;
  private diameterMaterial?: THREE.MeshBasicMaterial;
  private diameterColoringEnabled = true;
  private curveLine?: THREE.Line;
  private center = new THREE.Vector3();
  private extent = 160;
  private data?: Case;
  private labels: {
    element: HTMLButtonElement;
    point: THREE.Vector3;
    id: string;
  }[] = [];
  private raycaster = new THREE.Raycaster();
  private orbit = false;
  private flying = false;
  private flightProgress = 0.1;
  private flightTarget = 0.1;
  private flightPlaying = false;
  private flightSpeed = 0.018;
  private flightCurve?: THREE.CatmullRomCurve3;
  private flightUp = new THREE.Vector3(0, 1, 0);
  private flightUpInitialized = false;
  private heldKeys = new Set<string>();
  private branchProgress = new Map<string, number>();
  private wallRingRadii: number[] = [];
  private selected?: string;
  private labelsVisible = true;
  onFlightProgress: (progress: number) => void = () => {};
  onFlightPlaying: (playing: boolean) => void = () => {};
  onWallDiameterRange: (minMm: number, maxMm: number) => void = () => {};
  onCurrentDiameter: (mm: number) => void = () => {};

  constructor(
    private host: HTMLElement,
    private onSelect: (id: string) => void,
    private orientation: HTMLCanvasElement,
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0c1118, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.3;
    host.prepend(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 1400;
    this.controls.autoRotateSpeed = 0.6;
    this.scene.add(new THREE.AmbientLight(0xb5cbd7, 1.5));
    const key = new THREE.DirectionalLight(0xffe5df, 3.8);
    key.position.set(100, 200, 180);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x73dccc, 2.5);
    rim.position.set(-130, 60, -120);
    this.scene.add(rim);
    const fill = new THREE.DirectionalLight(0xd98985, 1.5);
    fill.position.set(0, -80, 150);
    this.scene.add(fill);
    this.scene.add(this.anatomy, this.branchObjects);
    let down = { x: 0, y: 0 };
    this.renderer.domElement.addEventListener("pointerdown", (event) => {
      down = { x: event.clientX, y: event.clientY };
    });
    this.renderer.domElement.addEventListener("pointerup", (event) => {
      if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5)
        return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.raycaster.setFromCamera(
        new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          (-(event.clientY - rect.top) / rect.height) * 2 + 1,
        ),
        this.camera,
      );
      const hit = this.raycaster.intersectObjects(
        this.branchObjects.children,
        true,
      )[0];
      if (hit?.object.userData.id)
        this.onSelect(hit.object.userData.id as string);
    });
    new ResizeObserver(() => this.resize()).observe(host);
    let previous = performance.now();
    this.renderer.setAnimationLoop((now) => {
      const delta = Math.min((now - previous) / 1000, 0.1);
      previous = now;
      if (this.flying) {
        if (this.flightPlaying) {
          this.flightProgress = Math.min(
            MAX_PROGRESS,
            this.flightProgress + delta * this.flightSpeed,
          );
          this.flightTarget = this.flightProgress;
          if (this.flightProgress >= MAX_PROGRESS) {
            this.flightPlaying = false;
            this.onFlightPlaying(false);
          }
          this.applyFlightPosition(this.flightProgress);
        } else {
          let direction = 0;
          if (this.heldKeys.has("ArrowUp")) direction += 1;
          if (this.heldKeys.has("ArrowDown")) direction -= 1;
          if (direction !== 0) {
            this.flightTarget = clamp(
              this.flightTarget + direction * ARROW_SPEED * delta,
              MIN_PROGRESS,
              MAX_PROGRESS,
            );
          }
          if (Math.abs(this.flightTarget - this.flightProgress) > 1e-4) {
            const ease = 1 - Math.pow(0.001, delta);
            this.applyFlightPosition(
              this.flightProgress +
                (this.flightTarget - this.flightProgress) * ease,
            );
          }
        }
      }
      if (!this.flying) this.controls.update();
      this.updateLabels();
      this.drawOrientation();
      this.renderer.render(this.scene, this.camera);
    });
  }

  private drawOrientation() {
    const context = this.orientation.getContext("2d")!;
    const inverse = this.camera.quaternion.clone().invert();
    context.clearRect(0, 0, 90, 90);
    context.font = "11px monospace";
    context.textAlign = "center";
    context.textBaseline = "middle";
    const axes = [
      {
        axis: new THREE.Vector3(1, 0, 0),
        labels: ["L", "R"],
        color: "#f4a899",
      },
      {
        axis: new THREE.Vector3(0, 0, -1),
        labels: ["P", "A"],
        color: "#71dbbd",
      },
      {
        axis: new THREE.Vector3(0, 1, 0),
        labels: ["S", "I"],
        color: "#a8baff",
      },
    ];
    for (const { axis, labels, color } of axes) {
      const p = axis.applyQuaternion(inverse);
      context.strokeStyle = color;
      context.fillStyle = color;
      context.beginPath();
      context.moveTo(45 - p.x * 24, 45 + p.y * 24);
      context.lineTo(45 + p.x * 24, 45 - p.y * 24);
      context.stroke();
      labels.forEach((label, i) => {
        const sign = i === 0 ? 1 : -1;
        context.fillText(label, 45 + p.x * 34 * sign, 45 - p.y * 34 * sign);
      });
    }
  }

  private resize() {
    const width = this.host.clientWidth,
      height = this.host.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  load(data: Case) {
    this.data = data;
    this.flying = false;
    this.flightPlaying = false;
    this.controls.enabled = true;
    this.anatomy.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((material) => material.dispose());
      }
    });
    this.anatomy.clear();
    const positions = new Float32Array(data.mesh.vertices.length);
    for (let i = 0; i < positions.length; i += 3) {
      positions[i] = data.mesh.vertices[i];
      positions[i + 1] = data.mesh.vertices[i + 2];
      positions[i + 2] = -data.mesh.vertices[i + 1];
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(data.mesh.faces);
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const bounds = geometry.boundingBox!;
    bounds.getCenter(this.center);
    this.extent = Math.max(...bounds.getSize(new THREE.Vector3()).toArray());
    this.flightCurve = new THREE.CatmullRomCurve3(
      data.centerline.map(viewPoint),
    );
    this.mapBranchesToProgress(data.branches);
    const wallColors = this.wallDiameterColors(positions);
    geometry.setAttribute("color", wallColors.attribute);
    this.wallRingRadii = wallColors.radiiByRing;
    this.onWallDiameterRange(
      wallColors.lowRadius * 2,
      wallColors.highRadius * 2,
    );
    this.tissueMaterial?.dispose();
    this.diameterMaterial?.dispose();
    this.tissueMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xb95854,
      roughness: 0.42,
      metalness: 0.08,
      clearcoat: 0.2,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
    });
    // Unlit on purpose: any lit material — no matter how matte — still
    // shades a neutral black/white surface with the scene's colored
    // ambient/rim lights, which showed up as an unwanted blue-ish tint.
    // MeshBasicMaterial ignores lighting entirely, so the ramp renders
    // exactly as specified at the cost of the wall's rounded shading.
    this.diameterMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
    });
    this.parent = new THREE.Mesh(
      geometry,
      this.diameterColoringEnabled ? this.diameterMaterial : this.tissueMaterial,
    );
    this.anatomy.add(this.parent);
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(
      data.centerline.map(viewPoint),
    );
    this.curveLine = new THREE.Line(
      lineGeometry,
      new THREE.LineBasicMaterial({
        color: 0xf6bcb0,
        transparent: true,
        opacity: 0.7,
        depthTest: false,
      }),
    );
    this.curveLine.visible = false;
    this.anatomy.add(this.curveLine);
    const grid = new THREE.GridHelper(
      this.extent * 1.4,
      16,
      0x26343e,
      0x19232d,
    );
    grid.position.copy(this.center);
    grid.position.y = bounds.min.y - 9;
    this.anatomy.add(grid);
    this.drawBranches(data.branches);
    this.reset();
  }

  /**
   * Colors the parent mesh by local vessel radius, one shared color per
   * "ring" (every vertex belonging to the same cross-section) rather than
   * per individual vertex — a single vertex's own nearest-centreline
   * distance is noisy (mesh irregularities, a stray branch-stub vertex),
   * which made vertex-level coloring look speckled instead of banded like
   * an actual cross-section. White = widest ring, black = thinnest.
   *
   * Each vertex is assigned to a ring by its closest point on the whole
   * centreline polyline (not just its nearest single sample point): at a
   * curve like the aortic arch, "nearest sample" snaps vertices on the
   * outside vs. inside of the bend to different, non-adjacent samples,
   * tearing one true ring into a jagged boundary. Closest-point-on-polyline
   * gives every vertex around the same true cross-section the same ring.
   *
   * The "widest point" reference excludes rings within a few rings of any
   * branch ostium: that's exactly where a daughter's stub is fused into the
   * parent surface, sitting far outside the aorta's own true radius. A
   * blanket percentile trim used to do this job, but cutting at the 90th
   * percentile was clipping a big share of the genuinely wide ascending
   * aorta/arch to flat white along with the real outliers. Excluding the
   * known branch locations directly lets the trim stay light (1st-99th
   * percentile, just a safety margin) while using the vessel's real range.
   */
  private wallDiameterColors(positions: Float32Array): {
    attribute: THREE.BufferAttribute;
    lowRadius: number;
    highRadius: number;
    radiiByRing: number[];
  } {
    const vertexCount = positions.length / 3;
    const samples = this.flightCurve!.getSpacedPoints(150);
    const vertex = new THREE.Vector3();
    const assignments = new Array<{ index: number; distance: number }>(
      vertexCount,
    );
    for (let i = 0; i < vertexCount; i++) {
      vertex.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      const { position, distance } = closestPolylinePosition(samples, vertex);
      assignments[i] = {
        index: clamp(Math.round(position), 0, samples.length - 1),
        distance,
      };
    }
    const radiiByRing = ringRadii(assignments, samples.length);
    const branchRings = [...this.branchProgress.values()].map((progress) =>
      ringIndexAt(progress, samples.length),
    );
    const excluded = ringsNear(branchRings, 3, samples.length);
    const [lowRadius, highRadius] = percentileRangeExcluding(
      radiiByRing,
      excluded,
      0.01,
      0.99,
    );
    const span = Math.max(highRadius - lowRadius, 1e-6);
    const ringColors = radiiByRing.map((radius) =>
      diameterColorRGB((radius - lowRadius) / span),
    );
    const colors = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i++) {
      const [r, g, b] = ringColors[assignments[i].index];
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
    return {
      attribute: new THREE.BufferAttribute(colors, 3),
      lowRadius,
      highRadius,
      radiiByRing,
    };
  }

  private mapBranchesToProgress(branches: Branch[]) {
    this.branchProgress.clear();
    if (!this.flightCurve) return;
    const samples = this.flightCurve.getSpacedPoints(200);
    for (const branch of branches) {
      const target = viewPoint(branch.ostium_xyz_mm);
      this.branchProgress.set(
        branch.instance_id,
        progressForPoint(samples, target),
      );
    }
  }

  private drawBranches(branches: Branch[]) {
    this.labels.forEach((label) => label.element.remove());
    this.labels = [];
    this.branchObjects.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((material) => material.dispose());
      }
    });
    this.branchObjects.clear();
    branches.forEach((branch, index) => {
      const color = COLORS[index % COLORS.length];
      const path = new THREE.CatmullRomCurve3(
        branch.path_xyz_mm.map(viewPoint),
      );
      const tube = new THREE.Mesh(
        new THREE.TubeGeometry(
          path,
          24,
          Math.max(branch.radius_mm * 0.62, 0.7),
          10,
          false,
        ),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.15,
          roughness: 0.3,
        }),
      );
      tube.userData.id = branch.instance_id;
      this.branchObjects.add(tube);
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(1.4, 16, 12),
        new THREE.MeshBasicMaterial({
          color: evidenceColor(branch.evidence_score),
          depthTest: false,
        }),
      );
      sphere.position.copy(viewPoint(branch.ostium_xyz_mm));
      sphere.userData.id = branch.instance_id;
      sphere.userData.evidenceScore = branch.evidence_score;
      sphere.renderOrder = 2;
      this.branchObjects.add(sphere);
      const direction = viewPoint(branch.direction_xyz).normalize();
      const arrow = new THREE.ArrowHelper(
        direction,
        viewPoint(branch.seed_xyz_mm),
        5,
        color,
        1.7,
        1,
      );
      this.branchObjects.add(arrow);
      const angle = branch.features.parent_angle_degrees;
      const angleText =
        typeof angle === "number" && Number.isFinite(angle)
          ? ` · ${Math.round(angle)}°`
          : "";
      const element = document.createElement("button");
      element.className = "branch-label";
      element.textContent = `${branch.instance_id.replace("branch_", "")}${angleText}`;
      element.style.setProperty("--branch-color", color);
      element.title = `Inspect ${branchName(branch.instance_id)}${angleText ? ` — angle to parent wall: ${Math.round(angle)}°` : ""}`;
      element.onclick = () => this.onSelect(branch.instance_id);
      this.host.append(element);
      this.labels.push({
        element,
        point: viewPoint(branch.seed_xyz_mm),
        id: branch.instance_id,
      });
    });
  }

  private updateLabels() {
    this.labels.forEach((label) => {
      const position = label.point.clone().project(this.camera);
      const visible =
        this.labelsVisible &&
        position.z < 1 &&
        position.z > -1 &&
        Math.abs(position.x) < 0.93 &&
        Math.abs(position.y) < 0.9;
      label.element.hidden = !visible;
      label.element.style.left = `${((position.x + 1) / 2) * this.host.clientWidth + 15}px`;
      label.element.style.top = `${((-position.y + 1) / 2) * this.host.clientHeight - 10}px`;
      label.element.classList.toggle("selected", label.id === this.selected);
    });
  }

  select(id: string) {
    this.selected = id;
  }

  focus(id: string) {
    const branch = this.data?.branches.find((b) => b.instance_id === id);
    if (!branch) return;
    if (this.flying) this.setFlythrough(false);
    const target = viewPoint(branch.ostium_xyz_mm);
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(new THREE.Vector3(25, 14, 55));
    this.controls.update();
  }

  reset() {
    if (this.flying) this.setFlythrough(false);
    this.camera.up.set(0, 1, 0);
    this.controls.target.copy(this.center);
    this.camera.position
      .copy(this.center)
      .add(
        new THREE.Vector3(
          this.extent * 0.38,
          this.extent * 0.08,
          this.extent * 1.93,
        ),
      );
    this.camera.near = 0.1;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  setOpacity(opacity: number) {
    if (this.tissueMaterial) this.tissueMaterial.opacity = opacity;
    if (this.diameterMaterial) this.diameterMaterial.opacity = opacity;
  }
  setParent(visible: boolean) {
    if (this.parent) this.parent.visible = visible;
  }
  setLabels(visible: boolean) {
    this.labelsVisible = visible;
  }
  setCenterline(visible: boolean) {
    if (this.curveLine) this.curveLine.visible = visible;
  }
  setDiameterColoring(enabled: boolean) {
    this.diameterColoringEnabled = enabled;
    if (!this.parent || !this.tissueMaterial || !this.diameterMaterial) return;
    this.parent.material = enabled ? this.diameterMaterial : this.tissueMaterial;
  }
  toggleOrbit() {
    this.orbit = !this.orbit;
    this.controls.autoRotate = this.orbit;
    return this.orbit;
  }
  zoom(factor: number) {
    this.camera.position
      .sub(this.controls.target)
      .multiplyScalar(factor)
      .add(this.controls.target);
  }

  setFlythrough(enabled: boolean) {
    this.flying = enabled;
    this.controls.enabled = !enabled;
    this.flightPlaying = false;
    this.heldKeys.clear();
    const flightOpacity = enabled ? 1 : 0.92;
    if (this.tissueMaterial) this.tissueMaterial.opacity = flightOpacity;
    if (this.diameterMaterial) this.diameterMaterial.opacity = flightOpacity;
    if (enabled) {
      this.flightUpInitialized = false;
      this.setFlightPosition(this.flightProgress);
    } else this.reset();
  }

  /** Immediate seek (slider drag, initial position, resuming after a jump). */
  setFlightPosition(progress: number) {
    this.flightTarget = progress;
    this.applyFlightPosition(progress);
  }

  /**
   * Positions the camera along the flight curve and keeps its orientation
   * roll-stable: rather than deriving "up" fresh from a fixed world axis
   * each frame (which flips/rolls whenever the centreline tangent swings
   * past vertical, as the aortic arch does), the up vector is parallel-
   * transported frame-to-frame so it only ever changes as much as the
   * tangent itself does.
   */
  private applyFlightPosition(progress: number) {
    this.flightProgress = progress;
    if (!this.flightCurve || !this.flying) return;
    const position = this.flightCurve.getPointAt(progress);
    const lookAt = this.flightCurve.getPointAt(Math.min(progress + 0.035, 1));
    const tangent = lookAt.clone().sub(position);
    if (tangent.lengthSq() < 1e-8)
      tangent.copy(this.flightCurve.getTangentAt(progress));
    tangent.normalize();
    if (!this.flightUpInitialized) {
      this.flightUp.copy(seedFlightUp(tangent));
      this.flightUpInitialized = true;
    } else {
      this.flightUp.copy(transportFlightUp(tangent, this.flightUp));
    }
    this.camera.position.copy(position);
    this.camera.up.copy(this.flightUp);
    this.camera.lookAt(lookAt);
    this.onFlightProgress(progress);
    if (this.wallRingRadii.length) {
      const ringIndex = ringIndexAt(progress, this.wallRingRadii.length);
      this.onCurrentDiameter(this.wallRingRadii[ringIndex] * 2);
    }
  }

  playFlight() {
    if (this.flightProgress >= MAX_PROGRESS) this.setFlightPosition(MIN_PROGRESS);
    this.flightPlaying = !this.flightPlaying;
    if (this.flightPlaying) this.heldKeys.clear();
    return this.flightPlaying;
  }
  setSpeed(speed: number) {
    this.flightSpeed = speed * 0.018;
  }

  /** Up/Down held: dt-integrated in the animation loop (see setAnimationLoop). */
  handleKeyDown(key: string) {
    if (!this.flying) return;
    if (key === "ArrowUp" || key === "ArrowDown") {
      this.flightPlaying = false;
      this.heldKeys.add(key);
    } else if (key === "ArrowLeft") {
      this.jumpToBranch(-1);
    } else if (key === "ArrowRight") {
      this.jumpToBranch(1);
    }
  }

  handleKeyUp(key: string) {
    this.heldKeys.delete(key);
  }

  private jumpToBranch(direction: 1 | -1) {
    if (!this.flying || this.branchProgress.size === 0) return;
    const next = nextBranchProgress(
      [...this.branchProgress.values()],
      this.flightTarget,
      direction,
    );
    if (next === undefined) return;
    this.flightPlaying = false;
    this.flightTarget = clamp(next, MIN_PROGRESS, MAX_PROGRESS);
  }
}
