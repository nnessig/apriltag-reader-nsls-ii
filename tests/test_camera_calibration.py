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


def run_calibration(images_dir, out):
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--images-dir", str(images_dir), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


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
    fs.release()


def test_empty_folder_exits_with_error(tmp_path):
    result = run_calibration(tmp_path, tmp_path / "calib.yaml")
    assert result.returncode == 1
    assert "No images found" in result.stderr


def test_too_few_good_frames_exits_with_error(tmp_path):
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "blank.png"), blank)

    result = run_calibration(tmp_path, tmp_path / "calib.yaml")

    assert result.returncode == 2
    assert "Only kept 0 good frames" in result.stderr


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
