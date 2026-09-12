"""Local Aorta Explorer: bounded CPU jobs, real CT volumes, and static web assets."""

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from threading import Lock
from urllib.parse import urlparse

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from detector import DetectorConfig, detect, detect_pool, parent_curve, physical_points, prepare_roi
from learning import FEATURE_NAMES, features as candidate_features
from nifti_io import read_nifti
import review_page

ROOT = Path(__file__).resolve().parent


@dataclass
class CaseData:
    metadata: dict
    ct: bytes
    mask: bytes


def build_case(
    image_path: Path, mask_path: Path, case_id: str, config: DetectorConfig | None = None
) -> CaseData:
    config = config or DetectorConfig()
    image = read_nifti(str(image_path))
    mask = read_nifti(str(mask_path))
    result = detect_pool(image, mask, config) if config.profile == "review" else detect(image, mask, config)
    if not np.any(sitk.GetArrayViewFromImage(mask) > 0):
        raise ValueError("This mask is empty; the CLI can export an empty prediction, but there is no aorta to view.")
    grid, parent = prepare_roi(image, mask, config)
    ct = np.clip(sitk.GetArrayFromImage(grid), -32768, 32767).astype("<i2")
    surface = ndi.gaussian_filter(np.pad(parent.astype(np.float32), 1), 0.65)
    vertices, faces, _, _ = marching_cubes(surface, level=0.5, step_size=2)
    vertices = physical_points(grid, vertices - 1)
    curve = parent_curve(parent, 1)
    centerline = physical_points(grid, curve)
    origin = grid.GetOrigin()
    basis = np.asarray(grid.GetDirection()).reshape(3, 3) @ np.diag(grid.GetSpacing())
    metadata = {
        "case_id": case_id,
        "size_xyz": grid.GetSize(),
        "origin_xyz": origin,
        "basis": basis.tolist(),
        "native_size_xyz": image.GetSize(),
        "native_spacing_xyz": image.GetSpacing(),
        "native_direction": image.GetDirection(),
        "aorta_volume_ml": round(float(parent.sum()) / 1000, 1),
        "coverage_mm": round(float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum()), 1),
        "mesh": {"vertices": np.round(vertices, 3).ravel().tolist(), "faces": faces.ravel().tolist()},
        "centerline": np.round(centerline, 3).tolist(),
        "profile": config.profile,
        "feature_names": FEATURE_NAMES,
        "branches": [{**asdict(b), "feature_vector": candidate_features(b)} for b in result.branches],
        "prediction": result.prediction(case_id),
        "diagnostics": result.diagnostics(),
    }
    return CaseData(metadata, ct.tobytes(), parent.astype(np.uint8).tobytes())


class CaseStore:
    def __init__(self, data_root: Path, config: DetectorConfig | None = None):
        self.data_root = data_root
        self.config = config or DetectorConfig()
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.cache: OrderedDict[str, CaseData] = OrderedDict()
        self.jobs: dict[str, dict] = {}

    def paths(self, case_id: str) -> tuple[Path, Path]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
            raise ValueError("Invalid case identifier.")
        directory = (self.data_root / case_id).resolve()
        if directory.parent != self.data_root.resolve():
            raise ValueError("Case must be inside the configured data directory.")
        images = sorted(directory.glob("orig*.nii*"))
        masks = sorted(directory.glob("mask*.nii*"))
        if len(images) != 1 or len(masks) != 1:
            raise ValueError("Expected one orig*.nii[.gz] and one mask*.nii[.gz] per case.")
        return images[0], masks[0]

    def list_cases(self) -> list[dict]:
        cases = []
        for directory in sorted(self.data_root.iterdir()):
            if not directory.is_dir():
                continue
            try:
                paths = self.paths(directory.name)
            except ValueError:
                continue
            ready = True
            for path in paths:
                with path.open("rb") as source:
                    ready &= not source.read(40).startswith(b"version https://git-lfs")
            cases.append({"id": directory.name, "available": ready})
        return cases

    def start(self, case_id: str) -> dict:
        paths = self.paths(case_id)
        with self.lock:
            if case_id in self.cache:
                self.cache.move_to_end(case_id)
                return {"status": "ready"}
            existing = self.jobs.get(case_id)
            if existing and existing["status"] in ("queued", "running"):
                return existing.copy()
            if sum(j["status"] in ("queued", "running") for j in self.jobs.values()) >= 3:
                raise ValueError("Analysis queue is full. Wait for the current cases to finish.")
            self.jobs[case_id] = {"status": "queued"}
        self.executor.submit(self._build, case_id, paths)
        return {"status": "queued"}

    def _build(self, case_id: str, paths: tuple[Path, Path]) -> None:
        with self.lock:
            self.jobs[case_id] = {"status": "running"}
        try:
            data = build_case(*paths, case_id, self.config)
            with self.lock:
                self.cache[case_id] = data
                self.cache.move_to_end(case_id)
                while len(self.cache) > 2:
                    evicted, _ = self.cache.popitem(last=False)
                    self.jobs.pop(evicted, None)
                self.jobs[case_id] = {"status": "ready"}
        except Exception as error:
            with self.lock:
                self.jobs[case_id] = {"status": "failed", "error": str(error)}

    def get(self, case_id: str) -> CaseData | None:
        with self.lock:
            case = self.cache.get(case_id)
            if case:
                self.cache.move_to_end(case_id)
            return case


def make_handler(
    store: CaseStore, static_root: Path, ledger: review_page.ReviewLedger | None = None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            if len(data) > 2048 and "gzip" in self.headers.get("Accept-Encoding", ""):
                data = gzip.compress(data, compresslevel=3)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def send_json(self, data: dict | list, status: int = 200) -> None:
            self.send_bytes(json.dumps(data, allow_nan=False).encode(), "application/json", status)

        def do_POST(self) -> None:
            if ledger and review_page.handle(self, store, ledger, "POST", urlparse(self.path).path):
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) != 4 or parts[:2] != ["api", "cases"] or parts[3] != "analyze":
                self.send_json({"error": "Unknown endpoint."}, 404)
                return
            # Browsers may start jobs only from this app's own origin.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.send_json({"error": "Cross-origin requests are not allowed."}, 403)
                return
            try:
                self.send_json(store.start(parts[2]), 202)
            except (OSError, ValueError) as error:
                self.send_json({"error": str(error)}, 400)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if ledger and review_page.handle(self, store, ledger, "GET", path):
                return
            if path == "/api/health":
                self.send_json({"status": "ok", "profile": store.config.profile})
                return
            if path == "/api/cases":
                self.send_json(store.list_cases())
                return
            parts = path.strip("/").split("/")
            if len(parts) in (3, 4) and parts[:2] == ["api", "cases"]:
                case_id = parts[2]
                if len(parts) == 4 and parts[3] == "status":
                    with store.lock:
                        job = store.jobs.get(case_id, {"status": "idle"}).copy()
                    self.send_json(job)
                    return
                case = store.get(case_id)
                if case is None:
                    self.send_json({"error": "Analyze this case first."}, 404)
                elif len(parts) == 3:
                    self.send_json(case.metadata)
                elif parts[3] == "ct":
                    self.send_bytes(case.ct, "application/octet-stream")
                elif parts[3] == "mask":
                    self.send_bytes(case.mask, "application/octet-stream")
                elif parts[3] == "prediction":
                    self.send_json(case.metadata["prediction"])
                else:
                    self.send_json({"error": "Unknown endpoint."}, 404)
                return
            if path.startswith("/api/"):
                self.send_json({"error": "Unknown endpoint."}, 404)
                return
            target = (static_root / path.lstrip("/")).resolve()
            if path == "/":
                target = static_root / "index.html"
            if not target.is_relative_to(static_root) or not target.is_file():
                self.send_json({"error": "Not found. Build the frontend with npm run build in web/."}, 404)
                return
            self.send_bytes(target.read_bytes(), mimetypes.guess_type(str(target))[0] or "application/octet-stream")

        def do_HEAD(self) -> None:
            self.do_GET()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Aorta Explorer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--review-mode", action="store_true",
        help="Run the loose review-profile detector so weak candidates reach human review.",
    )
    parser.add_argument(
        "--reviews", type=Path, default=ROOT / "labels" / "reviews.json",
        help="Where /review verdicts are saved (Explorer export schema).",
    )
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error("The data directory does not exist.")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    store = CaseStore(args.data_root, DetectorConfig.review() if args.review_mode else None)
    ledger = review_page.ReviewLedger(args.reviews)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, ROOT / "web" / "dist", ledger))
    print(f"Aorta Explorer listening on port {args.port} ({store.config.profile} detector profile)", flush=True)
    print(f"Slice review page: http://{args.host}:{args.port}/review  (verdicts -> {args.reviews})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
