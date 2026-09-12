import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Branch, Case, Point } from "./types";
import { branchName, COLORS } from "./types";

const viewPoint = (p: Point) => new THREE.Vector3(p[0], p[2], -p[1]);

export class AortaViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(35, 1, 0.1, 3000);
  private controls: OrbitControls;
  private anatomy = new THREE.Group();
  private branchObjects = new THREE.Group();
  private parent?: THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhysicalMaterial>;
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
  private flightPlaying = false;
  private flightSpeed = 0.018;
  private flightCurve?: THREE.CatmullRomCurve3;
  private selected?: string;
  private labelsVisible = true;
  onFlightProgress: (progress: number) => void = () => {};

  constructor(
    private host: HTMLElement,
    private onSelect: (id: string) => void,
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
      if (this.flying && this.flightPlaying) {
        this.flightProgress = Math.min(
          0.98,
          this.flightProgress + delta * this.flightSpeed,
        );
        if (this.flightProgress >= 0.98) this.flightPlaying = false;
        this.setFlightPosition(this.flightProgress);
      }
      if (!this.flying) this.controls.update();
      this.updateLabels();
      this.renderer.render(this.scene, this.camera);
    });
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
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xb95854,
      roughness: 0.42,
      metalness: 0.08,
      clearcoat: 0.2,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
    });
    this.parent = new THREE.Mesh(geometry, material);
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
    this.flightCurve = new THREE.CatmullRomCurve3(
      data.centerline.map(viewPoint),
    );
    this.drawBranches(data.branches);
    this.reset();
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
        new THREE.MeshBasicMaterial({ color, depthTest: false }),
      );
      sphere.position.copy(viewPoint(branch.ostium_xyz_mm));
      sphere.userData.id = branch.instance_id;
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
      const element = document.createElement("button");
      element.className = "branch-label";
      element.textContent = branch.instance_id.replace("branch_", "");
      element.style.setProperty("--branch-color", color);
      element.title = `Inspect ${branchName(branch.instance_id)}`;
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
    if (this.parent) this.parent.material.opacity = opacity;
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
    if (this.parent) {
      this.parent.material.side = THREE.DoubleSide;
      this.parent.material.opacity = enabled ? 1 : 0.92;
    }
    if (enabled) this.setFlightPosition(this.flightProgress);
    else this.reset();
  }

  setFlightPosition(progress: number) {
    this.flightProgress = progress;
    if (!this.flightCurve || !this.flying) return;
    this.camera.position.copy(this.flightCurve.getPointAt(progress));
    this.camera.lookAt(
      this.flightCurve.getPointAt(Math.min(progress + 0.035, 1)),
    );
    this.onFlightProgress(progress);
  }

  playFlight() {
    if (this.flightProgress >= 0.98) this.setFlightPosition(0.02);
    this.flightPlaying = !this.flightPlaying;
    return this.flightPlaying;
  }
  setSpeed(speed: number) {
    this.flightSpeed = speed * 0.018;
  }
}
