"""Slice-first candidate review pages served by the Explorer under /review; verdicts persist server-side."""

from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import re
from threading import Lock
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import SimpleITK as sitk

from learning import FEATURE_NAMES

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

if TYPE_CHECKING:
    from explorer import CaseData, CaseStore

HALF_MM = 15.0
STRIP_OFFSETS_MM = (-4, -2, 0, 2, 4)
SLAB_MM = 2.0
WINDOW = (-100.0, 600.0)
LABELS = ("confirmed", "rejected", "unreviewed")
CASE_ID = re.compile(r"[A-Za-z0-9_-]+")
_png_cache: dict[tuple[str, str, str], bytes] = {}
_render_lock = Lock()


def fingerprint(branch: dict) -> str:
    return json.dumps([
        branch["ostium_xyz_mm"], branch["seed_xyz_mm"], branch["direction_xyz"],
        branch["radius_mm"], branch["feature_vector"],
    ])


class ReviewLedger:
    """Verdict store in the same schema the Explorer exports, so learning.py reads it directly."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.rows: list[dict] = []
        if path.is_file():
            payload = json.loads(path.read_text())
            self.rows = [r for r in payload.get("records", []) if isinstance(r, dict)]

    def export(self) -> dict:
        return {
            "schema_version": 1, "feature_names": FEATURE_NAMES,
            "scope": "candidate_reviews_only", "records": self.rows,
        }

    def status(self, case_id: str, branch: dict) -> str:
        for row in self.rows:
            if row["case_id"] == case_id and row["instance_id"] == branch["instance_id"]:
                return str(row["label"]) if row.get("fingerprint") == fingerprint(branch) else "unreviewed"
        return "unreviewed"

    def set(self, case_id: str, branch: dict, label: str) -> None:
        if label not in LABELS:
            raise ValueError("Label must be confirmed, rejected or unreviewed.")
        with self.lock:
            rows = [
                r for r in self.rows
                if not (r["case_id"] == case_id and r["instance_id"] == branch["instance_id"])
            ]
            if label != "unreviewed":
                rows.append({
                    "case_id": case_id, "instance_id": branch["instance_id"], "label": label,
                    "features": [float(v) for v in branch["feature_vector"]],
                    "fingerprint": fingerprint(branch),
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                })
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            try:
                temporary.write_text(json.dumps(
                    {**self.export(), "records": rows}, indent=2, allow_nan=False
                ) + "\n")
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
            self.rows = rows

    def counts(self, case_id: str, branches: list[dict] | None = None) -> dict[str, int]:
        counts = {"confirmed": 0, "rejected": 0}
        if branches is not None:
            for branch in branches:
                status = self.status(case_id, branch)
                if status in counts:
                    counts[status] += 1
            return counts
        for row in self.rows:
            if row["case_id"] == case_id and row["label"] in counts:
                counts[row["label"]] += 1
        return counts


def _volumes(case: "CaseData") -> tuple[np.ndarray, np.ndarray]:
    size = case.metadata["size_xyz"]
    shape = (int(size[2]), int(size[1]), int(size[0]))
    ct = np.frombuffer(case.ct, dtype="<i2").reshape(shape).astype(np.float32)
    mask = np.frombuffer(case.mask, dtype=np.uint8).reshape(shape) > 0
    return ct, mask


def _to_index_zyx(meta: dict, point_mm: Any) -> np.ndarray:
    basis = np.asarray(meta["basis"], dtype=float)
    origin = np.asarray(meta["origin_xyz"], dtype=float)
    return np.linalg.solve(basis, np.asarray(point_mm, dtype=float) - origin)[::-1]


OPPOSITE = {"A": "P", "P": "A", "L": "R", "R": "L", "S": "I", "I": "S"}


def _orientation(meta: dict) -> dict[str, str]:
    basis = np.asarray(meta["basis"], dtype=float)
    direction = basis / np.linalg.norm(basis, axis=0)
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction.ravel().tolist())
    return dict(zip(("x_right", "y_up", "z_up"), orientation))


def _panel(ax: Any, image: np.ndarray, mask: np.ndarray, point: tuple[float, float] | None,
           path: np.ndarray | None, title: str, edges: tuple[str, str] | None = None,
           window: tuple[float, float] = WINDOW) -> None:
    ax.imshow(np.clip((image - window[0]) / (window[1] - window[0]), 0, 1), cmap="gray",
              origin="lower", vmin=0, vmax=1, interpolation="nearest")
    if mask.any():
        ax.contour(mask, levels=[0.5], colors="#48e0c0", linewidths=1.0, origin="lower")
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], "-", color="#ff5c5c", lw=1.8, solid_capstyle="round")
        ax.plot(path[-1, 0], path[-1, 1], "s", color="#ff5c5c", ms=4)
    if point is not None:
        ax.plot(point[0], point[1], "o", color="#ffd43b", ms=6, mec="black")
    if edges:
        up, right = edges
        style = dict(transform=ax.transAxes, color="#48e0c0", fontsize=9, fontweight="bold")
        ax.text(0.5, 0.98, up, ha="center", va="top", **style)
        ax.text(0.5, 0.02, OPPOSITE[up], ha="center", va="bottom", **style)
        ax.text(0.98, 0.5, right, ha="right", va="center", **style)
        ax.text(0.02, 0.5, OPPOSITE[right], ha="left", va="center", **style)
    ax.set_title(title, color="#c9d1d9", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def render_candidate(case: "CaseData", branch: dict) -> bytes:
    key = (str(case.metadata["case_id"]), str(branch["instance_id"]), fingerprint(branch))
    cached = _png_cache.get(key)
    if cached:
        return cached
    ct, mask = _volumes(case)
    meta = case.metadata
    spacing = float(np.linalg.norm(np.asarray(meta["basis"], dtype=float)[:, 0]))
    ost = _to_index_zyx(meta, branch["ostium_xyz_mm"])
    tip = _to_index_zyx(meta, np.asarray(branch["ostium_xyz_mm"]) + 8.0 * np.asarray(branch["direction_xyz"]))
    path = np.asarray([_to_index_zyx(meta, p) for p in branch.get("path_xyz_mm", [])], dtype=float)
    if len(path) < 2:
        path = np.vstack((ost, tip))
    half = max(4, int(round(HALF_MM / spacing)))
    centre = np.clip(np.round(ost).astype(int), 0, np.asarray(ct.shape) - 1)
    lo = np.maximum(centre - half, 0)
    hi = np.minimum(centre + half + 1, ct.shape)
    iz, iy, ix = (int(v) for v in centre)
    axes_of = _orientation(meta)
    mip_window = (100.0, 600.0)
    with _render_lock:
        fig = Figure(figsize=(16, 5.6), dpi=88)
        fig.patch.set_facecolor("#0d1117")
        grid = fig.add_gridspec(2, 20, hspace=0.28, wspace=0.18)
        # Whole-aorta locators: where along the aorta, and on which wall, this candidate sits.
        _panel(fig.add_subplot(grid[0, 0:4]), ct.max(axis=1), mask.max(axis=1), (ost[2], ost[0]),
               path[:, [2, 0]], "locator · acquisition XZ MIP", (axes_of["z_up"], axes_of["x_right"]), mip_window)
        _panel(fig.add_subplot(grid[1, 0:4]), ct.max(axis=2), mask.max(axis=2), (ost[1], ost[0]),
               path[:, [1, 0]], "locator · acquisition YZ MIP", (axes_of["z_up"], axes_of["y_up"]), mip_window)
        offset = np.asarray([lo[0], lo[1], lo[2]], dtype=float)
        local = path - offset
        # Thin-slab projections: a 1 mm vessel leaves any single slice at once, so show the brightest
        # voxel across ±SLAB_MM around each plane instead.
        slab = max(1, int(round(SLAB_MM / spacing)))
        z0, z1 = max(0, iz - slab), min(ct.shape[0], iz + slab + 1)
        y0, y1 = max(0, iy - slab), min(ct.shape[1], iy + slab + 1)
        x0, x1 = max(0, ix - slab), min(ct.shape[2], ix + slab + 1)
        title = f"±{SLAB_MM:g} mm slab at the origin"
        views = [
            (f"acquisition XY · {title}", ct[z0:z1, lo[1]:hi[1], lo[2]:hi[2]].max(axis=0), mask[iz, lo[1]:hi[1], lo[2]:hi[2]],
             (ost[2] - lo[2], ost[1] - lo[1]), local[:, [2, 1]], (axes_of["y_up"], axes_of["x_right"])),
            (f"acquisition XZ · {title}", ct[lo[0]:hi[0], y0:y1, lo[2]:hi[2]].max(axis=1), mask[lo[0]:hi[0], iy, lo[2]:hi[2]],
             (ost[2] - lo[2], ost[0] - lo[0]), local[:, [2, 0]], (axes_of["z_up"], axes_of["x_right"])),
            (f"acquisition YZ · {title}", ct[lo[0]:hi[0], lo[1]:hi[1], x0:x1].max(axis=2), mask[lo[0]:hi[0], lo[1]:hi[1], ix],
             (ost[1] - lo[1], ost[0] - lo[0]), local[:, [1, 0]], (axes_of["z_up"], axes_of["y_up"])),
        ]
        for column, (title, image, outline, point, trace, edges) in enumerate(views):
            start = 5 + column * 5
            _panel(fig.add_subplot(grid[0, start:start + 5]), image, outline, point, trace, title, edges)
        for column, offset_mm in enumerate(STRIP_OFFSETS_MM):
            z = int(np.clip(iz + round(offset_mm / spacing), 0, ct.shape[0] - 1))
            marker: tuple[float, float] | None = (ost[2] - lo[2], ost[1] - lo[1]) if offset_mm == 0 else None
            start = 5 + column * 3
            _panel(fig.add_subplot(grid[1, start:start + 3]), ct[z, lo[1]:hi[1], lo[2]:hi[2]],
                   mask[z, lo[1]:hi[1], lo[2]:hi[2]], marker, None, f"XY {offset_mm:+d} mm",
                   (axes_of["y_up"], axes_of["x_right"]))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    if len(_png_cache) > 400:
        _png_cache.clear()
    _png_cache[key] = buffer.getvalue()
    return _png_cache[key]


STYLE = """
body{margin:0;background:#0d1117;color:#c9d1d9;font:14px/1.45 system-ui,sans-serif}
a{color:#58a6ff;text-decoration:none} header{padding:14px 22px;border-bottom:1px solid #30363d;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header strong{font-size:17px} .pill{background:#161b22;border:1px solid #30363d;border-radius:999px;padding:3px 10px;font-size:12px}
main{padding:18px 22px;max-width:1240px} table{border-collapse:collapse} td,th{padding:6px 12px;border-bottom:1px solid #21262d;text-align:left}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;margin:0 0 18px;padding:14px;scroll-margin-top:70px}
.card.current{border-color:#58a6ff} .card img{width:100%;max-width:1150px;display:block;border-radius:6px;background:#000}
.row{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:10px 0 4px} .meta{color:#8b949e;font-size:12px}
button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:7px 14px;cursor:pointer;font-size:13px}
button.confirm{border-color:#2ea043} button.reject{border-color:#da3633} button.on.confirm{background:#2ea043;color:#fff}
button.on.reject{background:#da3633;color:#fff} .badge{font-weight:600} .badge.confirmed{color:#3fb950} .badge.rejected{color:#f85149}
.badge.unreviewed{color:#d29922} .hint{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:13px}
kbd{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:0 5px;font-size:12px}
"""


def _page(title: str, body: str) -> bytes:
    document = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<meta name='viewport' content='width=device-width'><style>{STYLE}</style></head><body>{body}"
        "<script>window.addEventListener('pageshow',e=>{"
        "if(e.persisted||performance.getEntriesByType('navigation')[0]?.type==='back_forward')location.reload();"
        "});</script></body></html>"
    )
    return document.encode()


def page_index(store: "CaseStore", ledger: ReviewLedger) -> bytes:
    rows = []
    for entry in store.list_cases():
        case_id = entry["id"]
        case = store.get(case_id)
        pool = len(case.metadata["branches"]) if case else None
        counts = ledger.counts(case_id, case.metadata["branches"] if case else None)
        pending = "" if pool is None else str(max(0, pool - counts["confirmed"] - counts["rejected"]))
        link = f"<a href='/review/{case_id}'>{case_id}</a>" if entry["available"] else f"{case_id} <span class='meta'>(LFS pointer)</span>"
        rows.append(
            f"<tr><td>{link}</td><td>{'' if pool is None else pool}</td><td>{counts['confirmed']}</td>"
            f"<td>{counts['rejected']}</td><td>{pending}</td></tr>"
        )
    body = (
        f"<header><strong>Branchseed review</strong><span class='pill'>{store.config.profile} detector profile</span>"
        f"<span class='pill'>{len(ledger.rows)} verdicts saved</span><a href='/review/export'>download reviews JSON</a>"
        f"<a href='/'>3D explorer</a></header><main>"
        "<div class='hint'>Open a case, judge every candidate from its CT slices, and the verdict is saved to disk immediately. "
        "Pool sizes appear once a case has been analysed.</div>"
        "<table><tr><th>case</th><th>pool</th><th>confirmed</th><th>rejected</th><th>pending</th></tr>"
        + "".join(rows) + "</table></main>"
    )
    return _page("Branchseed review", body)


def page_waiting(case_id: str, status: str) -> bytes:
    body = (
        f"<header><strong>Branchseed review</strong><a href='/review'>all cases</a></header><main>"
        f"<div class='hint'>Analysing <b>{html.escape(case_id)}</b> ({html.escape(status)}). This page refreshes on its own.</div></main>"
        "<meta http-equiv='refresh' content='3'>"
    )
    return _page(f"{case_id} · analysing", body)


def page_case(store: "CaseStore", ledger: ReviewLedger, case: "CaseData") -> bytes:
    case_id = str(case.metadata["case_id"])
    branches = case.metadata["branches"]
    cards = []
    for index, branch in enumerate(branches):
        status = ledger.status(case_id, branch)
        features = branch.get("features", {})
        warnings = " · ".join(html.escape(w) for w in branch.get("warnings", []))
        cards.append(
            f"<section class='card' id='{branch['instance_id']}' data-instance='{branch['instance_id']}' data-index='{index}'>"
            f"<div class='row'><strong>{branch['instance_id']}</strong><span class='badge {status}' data-badge>{status}</span>"
            f"<span class='meta'>radius {branch['radius_mm']:.2f} mm · evidence {branch['evidence_score']:.2f} · "
            f"HU vs blood {features.get('path_hu_relative', float('nan')):.2f} · bone {features.get('bone_distance_mm', float('nan')):.1f} mm · "
            f"angle {features.get('parent_angle_degrees', float('nan')):.0f}° · wall gap {features.get('connector_gap', float('nan')):.2f}</span></div>"
            f"<img loading='lazy' src='/review/{case_id}/{branch['instance_id']}.png' alt='CT crops for {branch['instance_id']}'>"
            f"<div class='row'><button class='confirm {'on' if status == 'confirmed' else ''}' data-label='confirmed'>Confirm</button>"
            f"<button class='reject {'on' if status == 'rejected' else ''}' data-label='rejected'>Reject</button>"
            f"<button data-label='unreviewed'>Clear</button><span class='meta'>{warnings}</span></div></section>"
        )
    counts = ledger.counts(case_id, branches)
    pending = max(0, len(branches) - counts["confirmed"] - counts["rejected"])
    body = (
        f"<header><strong>{html.escape(case_id)}</strong><span class='pill'>{len(branches)} candidates</span>"
        f"<span class='pill' id='progress'>{pending} pending</span><a href='/review'>all cases</a>"
        f"<a href='/#case={case_id}'>3D explorer</a><a href='/review/export'>download reviews JSON</a></header><main>"
        "<div class='hint'>Confirm when a bright tube leaves the green aorta outline at the yellow dot, along the red path, "
        "and keeps going for about 5 mm in at least one panel. The bottom row walks through consecutive axial slices around the origin. "
        "Views follow acquisition axes; edge letters indicate the nearest anatomical directions from the header. "
        "Keys: <kbd>c</kbd> confirm · <kbd>r</kbd> reject · <kbd>x</kbd> clear · <kbd>j</kbd>/<kbd>k</kbd> next/previous.</div>"
        + "".join(cards) + "</main>"
        "<script>"
        "const cards=[...document.querySelectorAll('.card')];let current=0;"
        "function focusCard(i){current=Math.max(0,Math.min(cards.length-1,i));cards.forEach((c,j)=>c.classList.toggle('current',j===current));"
        "cards[current].scrollIntoView({behavior:'smooth',block:'start'});}"
        "async function verdict(card,label){if(card.dataset.saving)return;card.dataset.saving='true';try{"
        "const r=await fetch(location.pathname+'/verdict',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({instance_id:card.dataset.instance,label})});if(!r.ok){alert('Verdict not saved: '+(await r.text()));return;}"
        "const data=await r.json();const badge=card.querySelector('[data-badge]');badge.textContent=label;badge.className='badge '+label;"
        "card.querySelectorAll('button[data-label]').forEach(b=>b.classList.toggle('on',b.dataset.label===label&&label!=='unreviewed'));"
        "document.getElementById('progress').textContent=data.pending+' pending';}"
        "catch(error){alert('Verdict not saved: '+error.message);}finally{delete card.dataset.saving;}}"
        "cards.forEach((card,i)=>{card.addEventListener('click',()=>{current=i;cards.forEach((c,j)=>c.classList.toggle('current',j===i));});"
        "card.querySelectorAll('button[data-label]').forEach(b=>b.addEventListener('click',e=>{e.stopPropagation();verdict(card,b.dataset.label);}));});"
        "document.addEventListener('keydown',e=>{if(!cards.length||e.ctrlKey||e.metaKey||e.altKey||e.repeat||e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();"
        "if(k==='j')focusCard(current+1);else if(k==='k')focusCard(current-1);"
        "else if(k==='c'){verdict(cards[current],'confirmed');focusCard(current+1);}else if(k==='r'){verdict(cards[current],'rejected');focusCard(current+1);}"
        "else if(k==='x')verdict(cards[current],'unreviewed');});"
        "if(cards.length)cards[0].classList.add('current');"
        "</script>"
    )
    return _page(f"{case_id} · review", body)


def handle(handler: Any, store: "CaseStore", ledger: ReviewLedger, method: str, path: str) -> bool:
    parts = path.strip("/").split("/")
    if parts[0] != "review":
        return False
    if method == "GET" and len(parts) == 1:
        handler.send_bytes(page_index(store, ledger), "text/html; charset=utf-8")
        return True
    if len(parts) == 1:
        handler.send_json({"error": "Unknown review endpoint."}, 404)
        return True
    if method == "GET" and parts[1] == "export":
        handler.send_bytes(json.dumps(ledger.export(), indent=2).encode(), "application/json")
        return True
    case_id = parts[1]
    if not CASE_ID.fullmatch(case_id):
        handler.send_json({"error": "Invalid case identifier."}, 404)
        return True
    case = store.get(case_id)
    if method == "GET" and len(parts) == 2:
        if case is None:
            try:
                status = store.start(case_id)["status"]
            except ValueError as error:
                handler.send_json({"error": str(error)}, 400)
                return True
            handler.send_bytes(page_waiting(case_id, status), "text/html; charset=utf-8")
        else:
            handler.send_bytes(page_case(store, ledger, case), "text/html; charset=utf-8")
        return True
    if case is None:
        handler.send_json({"error": "Analyse this case first."}, 404)
        return True
    if method == "GET" and len(parts) == 3 and parts[2].endswith(".png"):
        instance = parts[2][:-4]
        branch = next((b for b in case.metadata["branches"] if b["instance_id"] == instance), None)
        if branch is None:
            handler.send_json({"error": "Unknown candidate."}, 404)
        else:
            handler.send_bytes(render_candidate(case, branch), "image/png")
        return True
    if method == "POST" and len(parts) == 3 and parts[2] == "verdict":
        origin = handler.headers.get("Origin")
        if origin and origin.split("//", 1)[-1] != handler.headers.get("Host"):
            handler.send_json({"error": "Cross-origin requests are not allowed."}, 403)
            return True
        try:
            body = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0)) or 0) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Verdict must be a JSON object.")
            branch = next(b for b in case.metadata["branches"] if b["instance_id"] == body.get("instance_id"))
            ledger.set(case_id, branch, str(body.get("label")))
        except (ValueError, StopIteration, KeyError, TypeError) as error:
            handler.send_json({"error": f"Verdict rejected: {error}"}, 400)
            return True
        except OSError:
            handler.send_json({"error": "Verdict could not be saved to disk. Please retry."}, 500)
            return True
        counts = ledger.counts(case_id, case.metadata["branches"])
        pending = max(0, len(case.metadata["branches"]) - counts["confirmed"] - counts["rejected"])
        handler.send_json({**counts, "pending": pending})
        return True
    handler.send_json({"error": "Unknown review endpoint."}, 404)
    return True
