# Standalone judge download

The selected judge application is **fusion (score-before-merge)**. The user
selected its higher measured five-reference F1, 0.75676. The source ZIP and
Windows offline ZIP contain identical inference code and the bundled logistic
model. [Read the exact judge installation/run guide](START_HERE.md).

- [Windows x64 / Python 3.13 offline judge ZIP](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-judge-fusion-2026-09-13/branchseed-judge-fusion-windows-x64.zip)
- [Source-only judge ZIP](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-judge-fusion-2026-09-13/branchseed-judge-fusion-source.zip)
- [Separate predictions and verification evidence](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-judge-fusion-2026-09-13/branchseed-judge-fusion-evidence.zip)
- [Package validation and measured runtime](VERIFICATION.md)
- [Release assets and checksums](https://github.com/Coder-Meet/battleoftheschool/releases/tag/branchseed-judge-fusion-2026-09-13)
- [Explorer demo instructions](../DEMO_GUIDE.md)

The judge needs no Git clone, Git LFS, Node/npm, frontend, visualization
dependencies, research frameworks or training data. Python and the actual CT /
parent-mask pair are supplied separately.

The source ZIP is approximately **31 KB**; the offline Windows ZIP is
approximately **90.7 MB**, mostly scientific-library wheels. Matching
development predictions, three visual checks and resource measurements are
provided as a separate evidence asset on the release page.

## Repository layout

- `application/`: judge packaging, minimal pinned dependencies and run guide.
- `web/`: Explorer frontend; `explorer.py` is its Python server.
- Root Python modules: the shared detector implementation and development tools.

The packaging script copies only eight required source/model files and two
installation files. It verifies every runtime file against the validated source
hashes from commit `6e67aca527fac6eaa2e2f1dc97cea3a62a7b5677`. It does not move,
rewrite or duplicate the maintained detector implementation. Unrelated files,
model experiments, caches, screenshots and `node_modules` cannot enter the
application archive.

The separate historical strict release remains available. Do not submit that
ZIP in place of the selected **judge-fusion** application.

## Rebuild

Use Python 3.13.3 from the repository root. The builder itself uses only Python's
standard library and refuses to replace an existing output directory.

Source-only:

```bash
python application/build.py --output-dir outputs/judge-fusion-source
```

For both variants, first obtain the Windows wheel dependency closure while
connected. This cross-download also works from Linux:

```bash
python -m pip download --only-binary=:all: --platform win_amd64 \
  --python-version 3.13 --implementation cp --abi cp313 \
  -r application/requirements.txt --dest outputs/judge-windows-wheels
python application/build.py --output-dir outputs/judge-fusion \
  --windows-wheelhouse outputs/judge-windows-wheels
```

The existing final-release Windows wheelhouse can also be supplied; only the
ten pinned inference wheels are copied. No matplotlib or its extra dependencies
are included. ZIP entry ordering/timestamps and checksums are deterministic.
To adopt a changed algorithm, explicitly revalidate and update the builder's
source hashes first.
