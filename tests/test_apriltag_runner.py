import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import apriltag_runner as ar

TAG_ID = 7
TAG_PIXELS = 200
BORDER_PIXELS = 100


@pytest.fixture
def tag_image():
    """Synthetic grayscale image containing one tag36h11 marker with a white margin."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, TAG_ID, TAG_PIXELS)
    return cv2.copyMakeBorder(
        marker,
        BORDER_PIXELS,
        BORDER_PIXELS,
        BORDER_PIXELS,
        BORDER_PIXELS,
        cv2.BORDER_CONSTANT,
        value=255,
    )


@pytest.fixture(autouse=True)
def reset_pose_history():
    ar._POSE_HISTORY.clear()
    yield
    ar._POSE_HISTORY.clear()


def detect(gray, do_pose=False, tag_size=None, fx=None, fy=None, cx=None, cy=None):
    return ar.detect_tags(
        gray,
        families="tag36h11",
        nthreads=1,
        decimate=1.0,
        sigma=0.0,
        refine_edges=False,
        decode_sharpening=0.25,
        do_pose=do_pose,
        tag_size=tag_size,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


# ----------------------------
# Quaternion conversion
# ----------------------------


def test_quaternion_identity():
    assert ar.rotation_matrix_to_quaternion(np.eye(3)) == pytest.approx((0, 0, 0, 1))


def test_quaternion_90_degrees_about_z():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    s = np.sqrt(0.5)
    assert ar.rotation_matrix_to_quaternion(R) == pytest.approx((0, 0, s, s))


def test_quaternion_90_degrees_about_x():
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    s = np.sqrt(0.5)
    assert ar.rotation_matrix_to_quaternion(R) == pytest.approx((s, 0, 0, s))


def test_quaternion_is_unit_length():
    angle = np.radians(37.0)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert np.linalg.norm(ar.rotation_matrix_to_quaternion(R)) == pytest.approx(1.0)


# ----------------------------
# Preprocessing
# ----------------------------


def test_preprocess_converts_color_to_gray():
    bgr = np.zeros((20, 30, 3), dtype=np.uint8)
    gray = ar.preprocess(bgr, use_clahe=False, use_sharpen=False)
    assert gray.shape == (20, 30)


def test_preprocess_keeps_gray_input_unchanged():
    gray = np.full((20, 30), 100, dtype=np.uint8)
    out = ar.preprocess(gray, use_clahe=False, use_sharpen=False)
    assert np.array_equal(out, gray)
    assert out is not gray


def test_preprocess_upscale():
    gray = np.zeros((20, 30), dtype=np.uint8)
    out = ar.preprocess(gray, use_clahe=False, use_sharpen=False, upscale=2.0)
    assert out.shape == (40, 60)


def test_preprocess_clahe_and_sharpen_keep_shape_and_dtype():
    gray = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    out = ar.preprocess(gray, use_clahe=True, use_sharpen=True)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


def test_unsharp_mask_leaves_flat_image_unchanged():
    gray = np.full((32, 32), 100, dtype=np.uint8)
    assert np.array_equal(ar.unsharp_mask(gray), gray)


# ----------------------------
# Pose smoothing
# ----------------------------


def make_args(**kwargs):
    defaults = {"pose_window": 3, "pose_outlier_thresh": 0.0, "pose_average": "mean"}
    return SimpleNamespace(**{**defaults, **kwargs})


def test_filter_pose_window_of_one_returns_input():
    t = np.array([1.0, 2.0, 3.0])
    out = ar.filter_pose(1, t, make_args(pose_window=1))
    assert out is t


def test_filter_pose_mean():
    args = make_args()
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    out = ar.filter_pose(1, np.array([0.0, 0.0, 3.0]), args)
    assert out == pytest.approx([0.0, 0.0, 2.0])


def test_filter_pose_median():
    args = make_args(pose_average="median")
    for z in (1.0, 5.0, 2.0):
        out = ar.filter_pose(1, np.array([0.0, 0.0, z]), args)
    assert out == pytest.approx([0.0, 0.0, 2.0])


def test_filter_pose_last():
    args = make_args(pose_average="last")
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    out = ar.filter_pose(1, np.array([0.0, 0.0, 4.0]), args)
    assert out == pytest.approx([0.0, 0.0, 4.0])


def test_filter_pose_window_drops_old_samples():
    args = make_args(pose_window=2)
    for z in (1.0, 2.0, 3.0):
        out = ar.filter_pose(1, np.array([0.0, 0.0, z]), args)
    assert out == pytest.approx([0.0, 0.0, 2.5])


def test_filter_pose_rejects_outlier():
    args = make_args(pose_outlier_thresh=0.1)
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    assert ar.filter_pose(1, np.array([0.0, 0.0, 5.0]), args) is None


def test_filter_pose_tracks_tags_independently():
    args = make_args()
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    out = ar.filter_pose(2, np.array([0.0, 0.0, 9.0]), args)
    assert out == pytest.approx([0.0, 0.0, 9.0])


# ----------------------------
# Detection
# ----------------------------


def test_detects_synthetic_tag_id(tag_image):
    results = detect(tag_image)
    assert [d.tag_id for d in results] == [TAG_ID]


def test_no_detections_in_blank_image():
    blank = np.full((300, 300), 255, dtype=np.uint8)
    assert len(detect(blank)) == 0


def test_detector_is_cached(tag_image):
    ar._DETECTOR_CACHE.clear()
    detect(tag_image)
    detect(tag_image)
    assert len(ar._DETECTOR_CACHE) == 1


def test_pose_position_matches_pinhole_geometry(tag_image):
    # A tag that is TAG_PIXELS wide seen by a camera with focal length f is at
    # distance z = f * tag_size / TAG_PIXELS, centered on the principal point.
    f = 800.0
    size_m = 0.05
    center = (TAG_PIXELS + 2 * BORDER_PIXELS) / 2
    results = detect(tag_image, do_pose=True, tag_size=size_m, fx=f, fy=f, cx=center, cy=center)
    assert len(results) == 1
    x, y, z = results[0].pose_t.reshape(-1)
    assert z == pytest.approx(f * size_m / TAG_PIXELS, rel=0.1)
    assert abs(x) < 0.01
    assert abs(y) < 0.01


def test_pose_strings_format(tag_image):
    f = 800.0
    center = (TAG_PIXELS + 2 * BORDER_PIXELS) / 2
    (det,) = detect(tag_image, do_pose=True, tag_size=0.05, fx=f, fy=f, cx=center, cy=center)
    text = ar.pose_strings(det, None, 0.05)
    assert text.startswith(f"ID {TAG_ID}: t=[")
    assert "rpy=[" in text


# ----------------------------
# Drawing and file output
# ----------------------------


def test_draw_detections_returns_color_image(tag_image):
    results = detect(tag_image)
    out = ar.draw_detections(tag_image, results)
    assert out.shape == (*tag_image.shape, 3)


def test_canny_edges_finds_edges_of_tag(tag_image):
    assert ar.canny_edges(tag_image).any()


def test_write_debug_images_creates_expected_files(tmp_path, tag_image):
    base = str(tmp_path / "frame")
    edges = ar.canny_edges(tag_image)
    paths = ar.write_debug_images(
        base,
        tag_image,
        edges,
        ar.draw_detections(tag_image, []),
        ar.draw_edges_overlay(tag_image, edges),
    )
    assert [p.removeprefix(base) for p in paths] == [
        "_gray.png",
        "_overlay.png",
        "_edges_overlay.png",
    ]
    for p in paths:
        assert cv2.imread(p) is not None


# ----------------------------
# Command line
# ----------------------------


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apriltag_runner.py", "--image", "tag.png"])
    args = ar.parse_args()
    assert args.image == "tag.png"
    assert args.families == "tag36h11"
    assert args.tag_size is None
    assert args.frame_id == "camera"


def test_parse_args_requires_a_source(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apriltag_runner.py"])
    with pytest.raises(SystemExit):
        ar.parse_args()


def test_parse_args_rejects_both_sources(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["apriltag_runner.py", "--image", "a.png", "--cam-id", "DEV_1"]
    )
    with pytest.raises(SystemExit):
        ar.parse_args()
