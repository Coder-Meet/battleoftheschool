# Branchseed — Toralis Labs Challenge (Battle of the Schools)

Team repo for the Toralis Labs track: detect every artery that directly
branches off the abdominal aorta from a CT volume + aorta mask, and report
each one as a separate daughter instance.

Full problem statement: see the [Branchseed challenge doc](https://docs.google.com/document/d/1oRb2R9pauvsC-9hDIfr23ojLx90JpCt0jVZjCD5l5Cg/edit).

## Working agreement

**We are committing directly to `main`. There are no feature branches.**
Pull before you start working, commit small and often, push as soon as
something works. If you break `main`, fix it forward — don't force-push.

```bash
git pull
# ... do work ...
git add <files>
git commit -m "short description"
git pull --rebase   # pick up anyone else's commits
git push
```

Because everyone pushes to `main` directly, keep commits small and pull
often to minimize conflicts. If you hit a conflict, resolve it locally
before pushing — don't push broken code.

## What's in this repo

```
data/                   # 25 subjects, paired CT + aorta mask (tracked via Git LFS)
  subject001/
    orig1.nii           # CT volume
    mask1.nii           # binary aorta mask (1 = aorta, 0 = everything else)
  ...
run.py                  # entry point — see CLI contract below
requirements.txt
.gitattributes          # routes *.nii / *.nii.gz through Git LFS
```

The `.nii` files are large (the whole `data/` folder is ~2 GB). They are
tracked with **Git LFS**, not raw git. See setup below.

## Setup

1. Install Git LFS (one-time, per machine):
   ```bash
   brew install git-lfs      # macOS
   # or: https://git-lfs.com for other platforms
   git lfs install
   ```
2. Clone / pull as normal — LFS pointers resolve automatically once LFS is
   installed:
   ```bash
   git clone https://github.com/StevenTB1/battleoftheschool.git
   cd battleoftheschool
   ```
3. Python environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Run

The task spec requires the program to support this CLI contract:

```bash
python run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --output prediction.json
```

Output is one JSON file per case with the parent aorta and a list of
daughter branch instances (ostium, seed point, radius, direction). See the
challenge doc for the exact schema and definitions (ostium centre, daughter
seed, daughter radius, etc).

## Constraints to keep in mind while building

- No case-specific edits — the same code must run on all cases and the
  hidden eval set.
- No GPU. Target environment: 4 CPU cores, 8 GB RAM, no internet.
- Runtime target: ~60s/case average.
- Coordinates must be physical mm (use `SimpleITK.TransformIndexToPhysicalPoint`),
  not voxel indices.
- Don't try to name vessels anatomically — just `branch_001`, `branch_002`, ...
- Provide a visual check (aorta mask + detected ostia + direction arrows)
  for at least 3 cases before submission.

## Submission deadline

Sunday 11:00 AM. Have the repo pushed and a working demo before then.
