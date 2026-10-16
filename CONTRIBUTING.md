# Contributing to Vimba AprilTag Reader

Thanks for your interest in contributing to the Vimba AprilTag Reader. Bug
reports, fixes, documentation improvements, and new features are all
welcome.

Most pull requests only need a review from one other person before they
are merged. This guide covers where things live in the repo, how to set
up and test your changes locally, and what to expect during review.

## 1. What lives where

| Path | Purpose |
| --- | --- |
| `apriltag_runner.py` | AprilTag detection from images or an AVT camera (Vimba X), with optional pose estimation and ROS 2 publishing |
| `camera_calibration.py` | Charuco-based camera calibration |
| `check_container.py` | Sanity check for the container environment |
| `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` | ROS 2 Humble container setup |
| `requirements.txt` | Python dependencies |
| `calibration_images/` | Input images folder for calibration |
| `tag_images/` | Sample images for tag detection |
| `output/` | Generated output (calibration YAML, debug images) |
| `demo_resources/` | Demo material, see its own `README.md` |

Documentation lives in `README.md`. If you change a command-line flag,
a default, or the calibration board parameters, update the README in
the same pull request.

## 2. Fork and clone

Follow these instructions for
[forking a GitHub repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

If you want to work with a local copy on your workstation, use these
remotes:

- The `upstream` remote is `git@github.com:NSLS2/vimba-apriltag-reader.git`
- The `origin` remote is your fork, `git@github.com:YOUR-GITHUB-ID/vimba-apriltag-reader.git`

## 3. Create a branch

```bash
git checkout main
git pull upstream main
git checkout -b my-new-feature-branch
```

When you're ready to push for the first time:

```bash
git push -u origin my-new-feature-branch
```

The `-u` flag sets the upstream tracking branch, so later `git push`
commands need no extra arguments.

## 4. Set up a local environment

### Option A: Docker (recommended)

The container provides ROS 2 Humble and the Python dependencies.

```bash
docker compose build
docker compose run --rm apriltag-reader
```

The repository is mounted at `/workspace` inside the container, so
edits on the host appear immediately.

For camera mode, install the Allied Vision **Vimba X** SDK on the host
(default path `/opt/VimbaX`). `docker-compose.yml` mounts it read-only,
and `entrypoint.sh` installs the `vmbpy` wheel automatically if it is
found. Adjust the mount if your SDK lives elsewhere.

For preview windows, `DISPLAY` must be set on the host and the X11
socket must be reachable.

### Option B: Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`apriltag_runner.py` currently imports the ROS 2 packages (`rclpy`,
`std_msgs`, `geometry_msgs`, `tf2_ros`) at startup, so it needs a ROS 2
installation even for still images. Use Docker unless you have ROS 2
set up locally. `camera_calibration.py` only needs
`requirements.txt`. Camera mode also needs `vmbpy` from Vimba X.

## 5. Make and test your changes

Automated tests live in `tests/` and run in GitHub Actions on every pull
request, together with the lint checks. Changes that touch the camera or
ROS 2 publishing still need a manual check on real hardware, so say what
you ran in the PR description.

### Running the tests

```bash
pip install -r requirements.txt pytest
pytest
```

The tests cover the helper functions in `apriltag_runner.py`, tag
detection and pose on a synthetic tag, and `camera_calibration.py` on the
demo images. If ROS 2 is not installed, `tests/conftest.py` substitutes
mock ROS modules so the tests still run. Camera mode and ROS 2
publishing are not covered. If you have `nox` or `uv`, `nox -s tests`
and `nox -s lint` run the tests and lint checks in their own
environments.

### Linting and formatting

The repo uses [pre-commit](https://pre-commit.com/) with
[ruff](https://docs.astral.sh/ruff/) (lint and format), `codespell`,
and basic file checks. Settings live in `pyproject.toml` and
`.pre-commit-config.yaml`. Install the hooks once per clone:

```bash
pip install pre-commit
pre-commit install
```

The hooks then run on each `git commit`. To check files without
committing:

```bash
pre-commit run --all-files
```

Some hooks (`ruff-format`, `ruff-check --fix`, `end-of-file-fixer`,
`trailing-whitespace`) fix files in place. When that happens the commit
is aborted, so re-stage the changes and commit again.

Useful checks:

```bash
# Container environment sanity check
python3 check_container.py

# Still-image detection (no camera needed; add your own image to tag_images/)
python3 apriltag_runner.py --image tag_images/<your-image>.png --save-debug

# Calibration against the demo images
python3 camera_calibration.py --images-dir demo_resources/demo_calibration_images \
  --out output/camera_charuco_calib.yaml
```

Guidelines:

- Keep the camera (`vmbpy`) and pose-estimation features optional, so
  detection on still images works without a camera or intrinsics.
- Follow the existing style in the file you are editing. Use
  [PEP 8](https://peps.python.org/pep-0008/) for new code.
- Do not hard-code camera IDs, intrinsics, or absolute paths. Use
  command-line arguments.
- Add new dependencies to `requirements.txt` (and the `Dockerfile` if the
  container needs them).
- Do not commit large generated files. Calibration results and debug
  images belong in `output/` and should only be committed when they serve
  as a deliberate reference.
- Never commit credentials, keys, or tokens.

### Calibration changes

If you touch `camera_calibration.py`, remember the board geometry
(squares, square size, marker size, dictionary) must match the printed
board. Note any change to those defaults in both the PR and the README.
Always measure the printed board before trusting a calibration.

### Commit messages

Write a short imperative summary line (for example,
`Add --undistort flag to apriltag_runner`), and add a body if the why
isn't obvious.

When you are satisfied, push to your fork:

```bash
git push
```

## 6. Submit a pull request

[Submit your PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
to the `main` branch of `NSLS2/vimba-apriltag-reader`.

Please describe what you changed, why, and how you tested it
(image mode, camera mode, ROS 2 mode, calibration). The reviewer might
ask for changes to improve clarity, correctness, or style. Push the new
commits to the same branch and the PR updates automatically.

### PR submission checklist

Before clicking "Create pull request":

- [ ] The change runs in the mode(s) it affects (image, camera, ROS 2, calibration).
- [ ] Optional features (camera, pose) still degrade gracefully when their dependencies are absent.
- [ ] `README.md` is updated for any changed flags, defaults, or workflows.
- [ ] New dependencies are added to `requirements.txt` (and the `Dockerfile` if needed).
- [ ] `pre-commit run --all-files` and `pytest` pass.
- [ ] No secrets, large binaries, or stray output files are included.
- [ ] The PR description explains what changed and how it was tested.

## 7. After your PR is approved

A maintainer will merge your contribution. Thank you for contributing!
