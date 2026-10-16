# AGENTS.md — vimba-apriltag-reader

Per-repository agent instructions for the **Vimba AprilTag Reader**: a
Python tool that detects AprilTags in still images or from an Allied
Vision (AVT) camera via the Vimba X SDK, estimates 6-DoF poses, and can
publish results to ROS 2. It runs in a ROS 2 Humble container.

## Git policy

The agent must never create commits, tags, or push to remotes. The
user commits manually outside the agent session. Allowed:

- Read-only: `git status`, `git diff`, `git log`, `git show`,
  `git blame`, `git ls-files`
- Local staging: `git add` to prepare changes for the user's review
- File moves: `git mv` (does not create a commit)
- Branch ops without commits: `git switch`, `git restore`, `git stash`

Forbidden: `git commit`, `git push`, `git tag`, `git merge` with
merge commit, `git cherry-pick`, `git revert`, history rewrites.

### Commit attribution

When suggesting a commit message for work the agent materially
contributed to, end it with a `Co-Authored-By` trailer naming the
model, for example:

```text
Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
```

Do not guess the model. If the model is unknown, say so instead of
omitting the trailer. Put the trailer in the final `-m` of the
suggested command. See [AI_DISCLOSURE.md](AI_DISCLOSURE.md) for the
project's AI-use disclosure.

## Destructive operations

Ask the user before any destructive operation, including:

- `rm -rf` or any recursive delete
- `git clean -fd`, `git reset --hard`
- Force operations (`-f`, `--force`, `--force-with-lease`)
- Large refactors touching many files
- Anything that rewrites history or discards uncommitted work
- Deleting or overwriting files in `calibration_images/`,
  `tag_images/`, or `output/`. They are gitignored, so they cannot be
  recovered from git.

## Repository layout

| Path | Purpose |
| --- | --- |
| `apriltag_runner.py` | Main tool: tag detection, pose estimation, Vimba X camera capture, ROS 2 publishing |
| `camera_calibration.py` | Charuco camera calibration; writes intrinsics to YAML |
| `check_container.py` | Prints which required and optional modules import cleanly |
| `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` | ROS 2 Humble container |
| `requirements.txt` | Python deps (`numpy`, `opencv-python`, `opencv-contrib-python`, `pupil-apriltags`) |
| `calibration_images/`, `tag_images/`, `output/` | User data folders. Contents are gitignored; only `.gitkeep` is tracked |
| `demo_resources/` | Demo images and a README for the demo video walkthrough |
| `README.md`, `CONTRIBUTING.md`, `AI_DISCLOSURE.md` | Docs |

The README documents the CLI flags. Keep it in sync with
`parse_args()` in `apriltag_runner.py` and the arguments in
`camera_calibration.py`.

## Running and verifying

There is no test suite or CI yet. Verify changes by running the code.

Linting and formatting use ruff, configured in `pyproject.toml`, via
pre-commit (`.pre-commit-config.yaml`, which also runs `codespell` and
basic file-hygiene hooks). Run the checks on the files you changed
before suggesting a commit:

```bash
pre-commit run --files <path1> <path2> ...
```

Hooks like `ruff-format` and `ruff-check --fix` modify files in
place. Report which files a hook changed. If a hook fails to install
or run, report that and stop. Do not skip hooks with `SKIP=...`. The
`.pre-commit-config.yaml` file needs its `!` exception in `.gitignore`
because `*.yaml` is ignored.

- The ROS 2 Python packages (`rclpy`, `std_msgs`, `geometry_msgs`,
  `tf2_ros`) are imported unconditionally at the top of
  `apriltag_runner.py`. The script will not even import without ROS 2
  sourced. Run it in the container, where `entrypoint.sh` sources
  `/opt/ros/${ROS_DISTRO}/setup.bash`.
- `camera_calibration.py` needs only `numpy` and
  `opencv-contrib-python`, and can run outside the container.
- `pupil_apriltags` is required. `vmbpy` is optional and is detected at
  import time (`_vimba_available`). Camera mode exits with an error if
  it is missing.

```bash
docker compose build
docker compose run --rm apriltag-reader

# inside the container
python3 check_container.py
python3 apriltag_runner.py --image tag_images/<file>.png --save-debug
python3 camera_calibration.py \
    --images-dir demo_resources/demo_calibration_images \
    --out output/camera_charuco_calib.yaml
```

`tag_images/` is empty in a fresh clone, so supply an image yourself.
Camera mode needs real hardware and the Vimba X SDK mounted at
`/opt/VimbaX`. Do not claim camera or ROS behavior is verified unless
it was actually run. Say what was and was not exercised.

## Repo-specific knowledge

### Container

- Base image is `ros:humble-ros-base`. The repo is bind-mounted at
  `/workspace`, so host edits appear immediately without a rebuild.
  Rebuild only when `Dockerfile` or `requirements.txt` change.
- `Dockerfile` also `COPY`s the three Python scripts and
  `entrypoint.sh` into the image. A new top-level script must be added
  there for the image to contain it.
- The Vimba X SDK is installed on the host and mounted read-only at
  `/opt/VimbaX`. `entrypoint.sh` installs the `vmbpy` wheel from
  `/opt/VimbaX/api/python/` on startup if `vmbpy` is not importable.
- `docker-compose.yml` uses host networking and host IPC (needed for
  ROS 2 discovery) and forwards `DISPLAY` and `/tmp/.X11-unix` for
  `--preview` windows. The image is `vimba-apriltag-reader:humble`
  and the compose service is `apriltag-reader`.
- Container configuration is human-authored (see `AI_DISCLOSURE.md`).
  Propose changes to it rather than rewriting it.

### Gitignored paths

`.gitignore` excludes the contents of `output/`, `calibration_images/`
and `tag_images/`, plus `*.yaml`, `*.yml`, `*_gray.png`,
`*_overlay.png`, `*_edges_overlay.png`, and `*.log`. Consequences:

- Calibration YAML written to `output/` is never tracked.
- `docker-compose.yml` is tracked despite the `*.yml` rule. It was
  added before or alongside that rule. A **new** `.yml`/`.yaml` file
  needs `git add -f` or a `!` exception in `.gitignore`. Tell the user
  rather than force-adding silently.
- `demo_resources/demo_calibration_images/*.png` are tracked. Do not
  regenerate or reformat them.

### Calibration board

`camera_calibration.py` hard-codes the board: 5 × 3 squares, 5 mm
square size, 3.5 mm marker size, `DICT_4X4_50` (generated with
calib.io). The printed board must match. If these values change,
update the script, the README "Calibration Board" section, and say so
in the PR. A "0 good frames kept" failure usually means the squares
are too small to resolve, not a code bug. See the README's
troubleshooting section.

### Pose estimation and units

- `--tag-size` is in **meters**. `--fx/--fy/--cx/--cy` are in
  **pixels**. Pose is only computed when all of them are given.
- `camera_calibration.py` prints the `--fx/--fy/--cx/--cy` values to
  paste into `apriltag_runner.py`. Keep that output format stable, as
  the README shows it.
- The calibration YAML includes the distortion coefficients, but
  `apriltag_runner.py` does not read the YAML. Intrinsics are passed
  only via flags.

### ROS 2 interface

Published when `--ros` is set. Keep these stable, since downstream
consumers depend on them:

- `/apriltag_ids` — `std_msgs/Int32MultiArray`
- `/apriltag_poses` — `geometry_msgs/PoseArray`
- TF frames `<frame-id> -> tag_<id>` (`--frame-id`, default `camera`)

### Detector settings

`detect_tags` caches `pupil_apriltags.Detector` instances in
`_DETECTOR_CACHE` keyed by settings. Do not construct a new detector
per frame, as this is a performance optimization for the camera loop.

### Debug output names

`--save-debug` writes `<base>_gray.png`, `<base>_overlay.png`, and
`<base>_edges_overlay.png`, where `<base>` comes from `--out` or the
input image name. These suffixes are matched by `.gitignore`. If one
changes, update `.gitignore` and the README.

## Documentation conventions

- Markdown, written in a neutral, practical register. No jokes or
  stylized voice.
- The README is both user guide and calibration guide. Update it in
  the same change as any flag, default, or workflow change.
- Never put camera IDs, credentials, or machine-specific absolute
  paths into code or docs beyond the existing `DEV_...` example IDs.
- `AI_DISCLOSURE.md` lists models and tools used. If the user says
  they used a new tool or model, update its front matter and tables
  rather than appending prose elsewhere.
