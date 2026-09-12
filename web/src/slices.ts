import * as THREE from "three";
import type { Case, Point } from "./types";

const PLANES = [
  { name: "Axial", detail: "XY · acquisition plane", axes: [0, 1, 2] },
  { name: "Coronal", detail: "XZ · acquisition plane", axes: [0, 2, 1] },
  { name: "Sagittal", detail: "YZ · acquisition plane", axes: [1, 2, 0] },
];

export class SliceViews {
  private data?: Case;
  private ct = new Int16Array();
  private mask = new Uint8Array();
  private position: Point = [0, 0, 0];
  private inverse = new THREE.Matrix3();
  private overlay = true;
  private window = { level: 220, width: 700 };
  private canvases: HTMLCanvasElement[] = [];
  private sliders: HTMLInputElement[] = [];
  private buffers = PLANES.map(() => document.createElement("canvas"));
  private framePending = false;
  private frames: { x: number; y: number; w: number; h: number }[] = [];

  constructor(private host: HTMLElement) {
    host.innerHTML = PLANES.map(
      (plane, index) => `
      <article class="slice-card">
        <div class="slice-title"><span>${plane.name}</span><span class="slice-index">—</span></div>
        <div class="slice-canvas-wrap"><canvas aria-label="${plane.name} CT slice" data-index="${index}"></canvas>
          <span class="plane-axis">${plane.detail}</span></div>
        <input type="range" min="0" max="1" value="0" aria-label="${plane.name} slice" />
      </article>
    `,
    ).join("");
    this.canvases = [...host.querySelectorAll("canvas")];
    this.sliders = [...host.querySelectorAll("input")];
    this.sliders.forEach(
      (slider, i) =>
        (slider.oninput = () => {
          this.position[PLANES[i].axes[2]] = Number(slider.value);
          this.draw();
        }),
    );
    this.canvases.forEach((canvas, i) => {
      canvas.onclick = (event) => {
        if (!this.data) return;
        const rect = canvas.getBoundingClientRect(),
          frame = this.frames[i];
        if (!frame) return;
        const x = ((event.clientX - rect.left) * canvas.width) / rect.width;
        const y = ((event.clientY - rect.top) * canvas.height) / rect.height;
        if (
          x < frame.x ||
          x > frame.x + frame.w ||
          y < frame.y ||
          y > frame.y + frame.h
        )
          return;
        const [h, v] = PLANES[i].axes;
        this.position[h] = Math.round(
          ((x - frame.x) / frame.w) * (this.data.size_xyz[h] - 1),
        );
        this.position[v] = Math.round(
          (1 - (y - frame.y) / frame.h) * (this.data.size_xyz[v] - 1),
        );
        this.draw();
      };
      canvas.onwheel = (event) => {
        if (!this.data) return;
        event.preventDefault();
        const axis = PLANES[i].axes[2];
        this.position[axis] = Math.max(
          0,
          Math.min(
            this.data.size_xyz[axis] - 1,
            this.position[axis] + Math.sign(event.deltaY),
          ),
        );
        this.draw();
      };
    });
    new ResizeObserver(() => {
      if (this.framePending) return;
      this.framePending = true;
      requestAnimationFrame(() => {
        this.framePending = false;
        this.draw();
      });
    }).observe(host);
  }

  load(data: Case, ct: ArrayBuffer, mask: ArrayBuffer) {
    const voxels = data.size_xyz.reduce((total, size) => total * size, 1);
    if (ct.byteLength !== voxels * 2 || mask.byteLength !== voxels)
      throw new Error("Incomplete CT or mask download. Please retry.");
    this.data = data;
    this.ct = new Int16Array(ct);
    this.mask = new Uint8Array(mask);
    this.position = data.size_xyz.map((size) => Math.floor(size / 2)) as Point;
    const b = data.basis;
    this.inverse
      .set(
        b[0][0],
        b[0][1],
        b[0][2],
        b[1][0],
        b[1][1],
        b[1][2],
        b[2][0],
        b[2][1],
        b[2][2],
      )
      .invert();
    this.sliders.forEach(
      (slider, i) =>
        (slider.max = String(data.size_xyz[PLANES[i].axes[2]] - 1)),
    );
    this.draw();
  }

  focus(point: Point) {
    if (!this.data) return;
    const p = new THREE.Vector3(...point)
      .sub(new THREE.Vector3(...this.data.origin_xyz))
      .applyMatrix3(this.inverse);
    this.position = p
      .toArray()
      .map((value, i) =>
        Math.max(0, Math.min(this.data!.size_xyz[i] - 1, Math.round(value))),
      ) as Point;
    this.draw();
  }

  setOverlay(enabled: boolean) {
    this.overlay = enabled;
    this.draw();
  }
  setWindow(name: string) {
    this.window =
      name === "soft"
        ? { level: 60, width: 350 }
        : name === "bone"
          ? { level: 500, width: 1600 }
          : { level: 220, width: 700 };
    this.draw();
  }

  draw() {
    if (!this.data) return;
    const sizes = this.data.size_xyz;
    this.canvases.forEach((canvas, index) => {
      const [h, v, normal] = PLANES[index].axes;
      const width = sizes[h],
        height = sizes[v];
      const buffer = new Uint8ClampedArray(width * height * 4);
      const xyz = [...this.position];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          xyz[h] = x;
          xyz[v] = height - y - 1;
          const voxel = xyz[0] + sizes[0] * (xyz[1] + sizes[1] * xyz[2]);
          const value = Math.max(
            0,
            Math.min(
              255,
              ((this.ct[voxel] - this.window.level + this.window.width / 2) /
                this.window.width) *
                255,
            ),
          );
          const offset = (y * width + x) * 4;
          const mask = this.overlay && this.mask[voxel] > 0;
          buffer[offset] = mask ? value * 0.6 + 25 : value;
          buffer[offset + 1] = mask ? value * 0.6 + 85 : value;
          buffer[offset + 2] = mask ? value * 0.6 + 65 : value;
          buffer[offset + 3] = 255;
        }
      }
      if (!canvas.clientWidth || !canvas.clientHeight) return;
      const temporary = this.buffers[index];
      temporary.width = width;
      temporary.height = height;
      temporary
        .getContext("2d")!
        .putImageData(new ImageData(buffer, width, height), 0, 0);
      canvas.width = Math.max(
        1,
        canvas.clientWidth * Math.min(devicePixelRatio, 2),
      );
      canvas.height = Math.max(
        1,
        canvas.clientHeight * Math.min(devicePixelRatio, 2),
      );
      const context = canvas.getContext("2d")!;
      context.fillStyle = "#080d11";
      context.fillRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(canvas.width / width, canvas.height / height);
      const w = width * scale,
        ht = height * scale;
      const x = (canvas.width - w) / 2,
        y = (canvas.height - ht) / 2;
      this.frames[index] = { x, y, w, h: ht };
      context.drawImage(temporary, x, y, w, ht);
      const crossX = x + (this.position[h] + 0.5) * scale;
      const crossY = y + (height - this.position[v] - 0.5) * scale;
      context.strokeStyle = "#71dfc5";
      context.lineWidth = 1;
      context.setLineDash([5, 5]);
      context.beginPath();
      context.moveTo(crossX, y);
      context.lineTo(crossX, y + ht);
      context.moveTo(x, crossY);
      context.lineTo(x + w, crossY);
      context.stroke();
      context.setLineDash([]);
      context.strokeStyle = "#fbce7d";
      context.beginPath();
      context.arc(crossX, crossY, 4, 0, Math.PI * 2);
      context.stroke();
      this.sliders[index].value = String(this.position[normal]);
      this.host.querySelectorAll(".slice-index")[index].textContent =
        `${this.position[normal] + 1} / ${sizes[normal]}`;
    });
  }
}
