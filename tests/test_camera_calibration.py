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
