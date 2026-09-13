"""Build standalone fusion judge archives using only verified runtime files."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "6e67aca527fac6eaa2e2f1dc97cea3a62a7b5677"
SOURCE_HASHES = {
    "run.py": "c0e3d77d5296c75bc2308dac46cde986f7776915b359b0e02589e071af1b5edc",
    "detector.py": "9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e",
    "nifti_io.py": "d8cd2795046affe798a2c2ed0822dc615f73930b94381c1ef5af0de36127cb02",
    "learning.py": "d7acdaa6f78ea9f3b6703d0b457880de37804ee0fc06ff6aeb4c9bfe7fe82c1e",
    "origin_recovery.py": "1bddeefff79c881cef0e0678404d6746467ae3649f3db3ea5c35ad7f439ff1c2",
    "pipeline.py": "ebf9d0ab8ba903e401a5c930d899c5d71cb36229c34ce2efe4bf08fd501582c3",
    "batch.py": "92fc2c956b71c6411bfd42bac0be8f4e02caed1a8680e2165bb83f6588bfb9b2",
    "models/production-v1/logistic.json": "e9f864956aa65c3f05c038b13b0c8ca9e7289e4a44f6f0eb7837ab85814c730e",
}
BUNDLE_NAME = "branchseed-judge-fusion"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runtime_files(root: Path) -> dict[str, bytes]:
    files = {}
    for name, expected in SOURCE_HASHES.items():
        content = (root / name).read_bytes()
        if digest(content) != expected:
            raise ValueError(f"Runtime differs from validated source: {name}. Revalidate before updating pins.")
        files[name] = content
    for name in ("requirements.txt", "START_HERE.md"):
        files[name] = (root / "application" / name).read_bytes()
    return files


def supported_wheel(path: Path) -> bool:
    _, python, abi, platform = path.stem.rsplit("-", 3)
    if platform == "any":
        return abi == "none" and "py3" in python.split(".")
    if platform != "win_amd64":
        return False
    stable_abi = (
        python.startswith("cp3") and python[3:].isdigit()
        and 2 <= int(python[3:]) <= 13 and abi == "abi3"
    )
    return (python == "cp313" and abi == "cp313") or stable_abi


def windows_wheels(directory: Path, requirements: bytes) -> dict[str, bytes]:
    files = {}
    for line in requirements.decode("utf-8").splitlines():
        pin = line.partition("#")[0].strip()
        if not pin:
            continue
        parts = pin.split("==")
        if len(parts) != 2:
            raise ValueError(f"Expected a pinned requirement: {pin}")
        name, version = parts
        matches = [
            path for path in directory.glob(f"{name.lower().replace('-', '_')}-{version}-*.whl")
            if supported_wheel(path)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one Windows/universal wheel for {pin}, found {len(matches)}.")
        path = matches[0]
        files[f"windows-wheelhouse/{path.name}"] = path.read_bytes()
    return files


def write_bundle(destination: Path, files: dict[str, bytes], variant: str) -> None:
    manifest = {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "pipeline": "score-before-merge",
        "native_contrast_scale": 0.9,
        "candidate_threshold": 0.15,
        "python": "3.13.3",
        "variant": variant,
        "files_sha256": {name: digest(content) for name, content in sorted(files.items())},
    }
    payload = {**files, "MANIFEST.json": (json.dumps(manifest, indent=2) + "\n").encode("utf-8")}
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(payload.items()):
            entry = ZipInfo(f"{BUNDLE_NAME}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, content, compresslevel=9)


def build(root: Path, output: Path, wheelhouse: Path | None = None) -> list[Path]:
    files = runtime_files(root)
    wheels = windows_wheels(wheelhouse, files["requirements.txt"]) if wheelhouse is not None else None
    output.mkdir(parents=True, exist_ok=False)
    artifacts = [output / f"{BUNDLE_NAME}-source.zip"]
    write_bundle(artifacts[0], files, "source")
    if wheels is not None:
        artifacts.append(output / f"{BUNDLE_NAME}-windows-x64.zip")
        write_bundle(artifacts[-1], {**files, **wheels}, "windows-x64-cpython313")
    checksums = "".join(f"{digest(path.read_bytes())}  {path.name}\n" for path in artifacts)
    (output / "judge-fusion-SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrite a build.")
    parser.add_argument("--windows-wheelhouse", type=Path, help="Optional CPython 3.13 x64 wheel directory.")
    args = parser.parse_args()
    try:
        for artifact in build(ROOT, args.output_dir, args.windows_wheelhouse):
            print(f"{artifact}: {artifact.stat().st_size:,} bytes")
    except (OSError, ValueError) as error:
        print(f"Judge packaging: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
