import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "camera_calibration.py"
DEMO_IMAGES = ROOT / "demo_resources" / "demo_calibration_images"


# Standardizes how the script is run in all downstream tests and that non-zero exit codes don't crash pytest
def run_calibration(images_dir, out):
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--images-dir", str(images_dir), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


# Validates the entire calibration pipeline end-to-end
def test_calibration_on_demo_images(tmp_path):
    out = tmp_path / "calib.yaml"
    result = run_calibration(DEMO_IMAGES, out)

    assert result.returncode == 0, result.stderr
    match = re.search(r"Kept frames: (\d+) / (\d+)", result.stdout)
    assert match, result.stdout
    assert int(match.group(1)) >= 10

    # The printed intrinsics are what users paste into apriltag_runner.py.
    assert re.search(r"--fx \S+ --fy \S+ --cx \S+ --cy \S+", result.stdout)

    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    assert camera_matrix.shape == (3, 3)
    assert camera_matrix[0, 0] > 0
    assert camera_matrix[1, 1] > 0
    assert fs.getNode("image_width").real() > 0
    assert fs.getNode("image_height").real() > 0
    assert fs.getNode("rms").real() > 0
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    assert dist_coeffs is not None
    assert dist_coeffs.size >= 4
    fs.release()


# Enforces proper fail-fast behavior instead of silently producing junk files
def test_empty_folder_exits_with_error(tmp_path):
    result = run_calibration(tmp_path, tmp_path / "calib.yaml")
    assert result.returncode == 1
    assert "No images found" in result.stderr


# Checks that non-image files do not crash calibration
def test_non_image_files_are_ignored(tmp_path):
    (tmp_path / "readme.txt").write_text("hello world")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")

    result = run_calibration(tmp_path, tmp_path / "calib.yaml")

    assert result.returncode == 1
    assert "No images found" in result.stderr


# Ensures the script distinguishes between no input images, and images that exist but are unsuitable for calibration
def test_too_few_good_frames_exits_with_error(tmp_path):
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "blank.png"), blank)

    result = run_calibration(tmp_path, tmp_path / "calib.yaml")

    assert result.returncode == 2
    assert "Only kept 0 good frames" in result.stderr


# Checks that the --min-markers threshold is enforced
def test_min_markers_threshold_rejects_frames(tmp_path):
    # Create a completely blank image that cannot contain any calibration markers
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "blank.png"), blank)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--images-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "calib.yaml"),
            "--min-markers",
            "50",  # impossible threshold
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert "Only kept" in result.stderr


# Ensures that when tests are run that the printed text includes all documented CLI flags
@pytest.mark.parametrize("flag", ["--images-dir", "--out", "--show", "--min-markers"])
def test_help_lists_documented_flags(flag):
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert flag in result.stdout


# Renders the script's Charuco board (5x3, 5 mm squares, DICT_4X4_50) through a known
# pinhole camera from several poses, so the recovered intrinsics can be checked
# against ground truth instead of only checking that the script runs.
TRUE_K = (1500.0, 1500.0, 640.0, 480.0)  # fx, fy, cx, cy
IMAGE_SIZE = (1280, 960)
SQUARE_M = 0.005


def make_synthetic_calibration_images(out_dir, count=16, seed=0):
    rng = np.random.default_rng(seed)
    fx, fy, cx, cy = TRUE_K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 3), SQUARE_M, 0.0035, dictionary)
    tex = board.generateImage((1000, 600), marginSize=0, borderBits=1)
    th, tw = tex.shape
    half_w, half_h = 5 * SQUARE_M / 2, 3 * SQUARE_M / 2
    obj = np.float32(
        [[-half_w, -half_h, 0], [half_w, -half_h, 0], [half_w, half_h, 0], [-half_w, half_h, 0]]
    )
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    for i in range(count):
        rvec = np.array(
            [
                rng.uniform(-0.6, 0.6),
                rng.uniform(-0.6, 0.6),
                rng.uniform(-0.5, 0.5),
            ]
        )
        tvec = np.array(
            [rng.uniform(-0.005, 0.005), rng.uniform(-0.004, 0.004), rng.uniform(0.09, 0.14)]
        )
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        H = cv2.getPerspectiveTransform(src, pts.reshape(4, 2).astype(np.float32))
        img = cv2.warpPerspective(tex, H, IMAGE_SIZE, flags=cv2.INTER_AREA, borderValue=200)
        cv2.imwrite(str(out_dir / f"synthetic_{i:02d}.png"), img)


# Recovered intrinsics must match the camera the images were rendered with
def test_calibration_recovers_known_intrinsics(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    make_synthetic_calibration_images(images)
    out = tmp_path / "calib.yaml"

    result = run_calibration(images, out)
    assert result.returncode == 0, result.stderr

    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    rms = fs.getNode("rms").real()
    fs.release()

    fx, fy, cx, cy = TRUE_K
    assert K[0, 0] == pytest.approx(fx, rel=0.03)
    assert K[1, 1] == pytest.approx(fy, rel=0.03)
    # The principal point is much less well constrained than focal length: the script
    # fits free distortion from a small board, which trades off against cx/cy. Synthetic
    # runs land 20-40 px off, so this only guards against gross errors.
    assert K[0, 2] == pytest.approx(cx, abs=60)
    assert K[1, 2] == pytest.approx(cy, abs=60)
    assert rms < 1.0
