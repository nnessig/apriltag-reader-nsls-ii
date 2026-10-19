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


def detect(gray, do_pose=False, tag_size=None, fx=None, fy=None, cx=None, cy=None, **kwargs):
    return ar.detect_tags(
        gray,
        families="tag36h11",
        nthreads=1,
        decimate=kwargs.get("decimate", 1.0),
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


# Sends the identity rotation matrix (no rotation) and expects back the identity quaternion
def test_quaternion_identity():
    assert ar.rotation_matrix_to_quaternion(np.eye(3)) == pytest.approx((0, 0, 0, 1))


# Checks that the function handles single-axis rotations about z
def test_quaternion_90_degrees_about_z():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    s = np.sqrt(0.5)
    assert ar.rotation_matrix_to_quaternion(R) == pytest.approx((0, 0, s, s))


# Checks that the function handles single-axis rotations about x
def test_quaternion_90_degrees_about_x():
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    s = np.sqrt(0.5)
    assert ar.rotation_matrix_to_quaternion(R) == pytest.approx((s, 0, 0, s))


# Checks that quaternions are normalized
def test_quaternion_is_unit_length():
    angle = np.radians(37.0)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert np.linalg.norm(ar.rotation_matrix_to_quaternion(R)) == pytest.approx(1.0)


# ----------------------------
# Preprocessing
# ----------------------------


# Ensures the image is correctly converted to grayscale and retains the same dimensions
def test_preprocess_converts_color_to_gray():
    bgr = np.zeros((20, 30, 3), dtype=np.uint8)
    gray = ar.preprocess(bgr, use_clahe=False, use_sharpen=False)
    assert gray.shape == (20, 30)


# Checks that while the object is changed, the image itself remains the same
def test_preprocess_keeps_gray_input_unchanged():
    gray = np.full((20, 30), 100, dtype=np.uint8)
    out = ar.preprocess(gray, use_clahe=False, use_sharpen=False)
    assert np.array_equal(out, gray)
    assert out is not gray


# Tests that upscale produces the expected output
def test_preprocess_upscale():
    gray = np.zeros((20, 30), dtype=np.uint8)
    out = ar.preprocess(gray, use_clahe=False, use_sharpen=False, upscale=2.0)
    assert out.shape == (40, 60)


# Tests that downscale produces the expected output
def test_preprocess_downscale():
    img = np.zeros((40, 60), dtype=np.uint8)
    out = ar.preprocess(img, use_clahe=False, use_sharpen=False, upscale=0.5)
    assert out.shape == (20, 30)


# Ensures that CLAHE and sharpening does not change dimensions or dtype
def test_preprocess_clahe_and_sharpen_keep_shape_and_dtype():
    gray = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    out = ar.preprocess(gray, use_clahe=True, use_sharpen=True)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


# Verifies stability of the sharpening filter
def test_unsharp_mask_leaves_flat_image_unchanged():
    gray = np.full((32, 32), 100, dtype=np.uint8)
    assert np.array_equal(ar.unsharp_mask(gray), gray)


# ----------------------------
# Pose smoothing
# ----------------------------


# Utility for building argument objects with defaults
def make_args(**kwargs):
    defaults = {"pose_window": 3, "pose_outlier_thresh": 0.0, "pose_average": "mean"}
    return SimpleNamespace(**{**defaults, **kwargs})


# Ensures the optimization path works
def test_filter_pose_window_of_one_returns_input():
    t = np.array([1.0, 2.0, 3.0])
    out = ar.filter_pose(1, t, make_args(pose_window=1))
    assert out is t


# Confirms mean averaging
def test_filter_pose_mean():
    args = make_args()
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    out = ar.filter_pose(1, np.array([0.0, 0.0, 3.0]), args)
    assert out == pytest.approx([0.0, 0.0, 2.0])


# Checks the median output is correct
def test_filter_pose_median():
    args = make_args(pose_average="median")
    for z in (1.0, 5.0, 2.0):
        out = ar.filter_pose(1, np.array([0.0, 0.0, z]), args)
    assert out == pytest.approx([0.0, 0.0, 2.0])


# Checks the correctness of the last pose and ability to modify it
def test_filter_pose_last():
    args = make_args(pose_average="last")
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    out = ar.filter_pose(1, np.array([0.0, 0.0, 4.0]), args)
    assert out == pytest.approx([0.0, 0.0, 4.0])


# Checks that only the most recent samples remain after removing old ones
def test_filter_pose_window_drops_old_samples():
    args = make_args(pose_window=2)
    for z in (1.0, 2.0, 3.0):
        out = ar.filter_pose(1, np.array([0.0, 0.0, z]), args)
    assert out == pytest.approx([0.0, 0.0, 2.5])


# Tests outlier rejection
def test_filter_pose_rejects_outlier():
    args = make_args(pose_outlier_thresh=0.1)
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    assert ar.filter_pose(1, np.array([0.0, 0.0, 5.0]), args) is None


# Shows that the pose history is tag-ID specific
def test_filter_pose_tracks_tags_independently():
    args = make_args()
    ar.filter_pose(1, np.array([0.0, 0.0, 1.0]), args)
    out = ar.filter_pose(2, np.array([0.0, 0.0, 9.0]), args)
    assert out == pytest.approx([0.0, 0.0, 9.0])


# ----------------------------
# Detection
# ----------------------------


# Detects a synthetic AprilTag and asserts exactly one detection with the correct ID
def test_detects_synthetic_tag_id(tag_image):
    results = detect(tag_image)
    assert [d.tag_id for d in results] == [TAG_ID]


# Verifies multi-tag detection correctness
def test_detects_multiple_tags():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    tag1 = cv2.aruco.generateImageMarker(dictionary, 1, 150)
    tag2 = cv2.aruco.generateImageMarker(dictionary, 2, 150)

    canvas = np.full((450, 450), 255, dtype=np.uint8)
    canvas[50:200, 50:200] = tag1
    canvas[230:380, 230:380] = tag2

    results = detect(canvas)
    ids = sorted([d.tag_id for d in results])

    assert ids == [1, 2]


# Ensures that a blank white image produces no detections
def test_no_detections_in_blank_image():
    blank = np.full((300, 300), 255, dtype=np.uint8)
    assert len(detect(blank)) == 0


# Confirms the internal detector creation is cached and only one instance is created
def test_detector_is_cached(tag_image):
    ar._DETECTOR_CACHE.clear()
    detect(tag_image)
    detect(tag_image)
    assert len(ar._DETECTOR_CACHE) == 1


# Test that the detector cache invalidates when parameters change
def test_detector_cache_changes_with_params(tag_image):
    ar._DETECTOR_CACHE.clear()
    detect(tag_image)  # default settings
    detect(tag_image, decimate=2.0)  # changed detector parameter
    assert len(ar._DETECTOR_CACHE) == 2


# Checks the geometry, that the translation is consistent with camera params and that the tag is centered
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


# Checks that the tag detection rotates with the image
def test_pose_rotates_with_image(tag_image):
    rotated = cv2.rotate(tag_image, cv2.ROTATE_90_CLOCKWISE)

    det1 = detect(tag_image, do_pose=True, tag_size=0.05, fx=800, fy=800, cx=200, cy=200)[0]
    det2 = detect(rotated, do_pose=True, tag_size=0.05, fx=800, fy=800, cx=200, cy=200)[0]

    rpy1 = det1.pose_rpy
    rpy2 = det2.pose_rpy
    delta_yaw = abs(rpy2[2] - rpy1[2])

    assert delta_yaw == pytest.approx(np.pi / 2, rel=0.2)


# Verifies the presence and formatting of expected fields
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


# Checks that a 3-channel color image is returned
def test_draw_detections_returns_color_image(tag_image):
    results = detect(tag_image)
    out = ar.draw_detections(tag_image, results)
    assert out.shape == (*tag_image.shape, 3)


# Runs Canny on the tag image and expects at least some edges found
def test_canny_edges_finds_edges_of_tag(tag_image):
    assert ar.canny_edges(tag_image).any()


# Checks that the expected files are written (_gray.png, _overlay.png, _edges_overlay.png) and confirms they are readable
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


# Ensures default options are correctly populated (image, families, tag_size, frame_id)
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apriltag_runner.py", "--image", "tag.png"])
    args = ar.parse_args()
    assert args.image == "tag.png"
    assert args.families == "tag36h11"
    assert args.tag_size is None
    assert args.frame_id == "camera"


# Ensures CLI enforces one required input source
def test_parse_args_requires_a_source(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apriltag_runner.py"])
    with pytest.raises(SystemExit):
        ar.parse_args()


# Ensures mutual exclusion in CLI options
def test_parse_args_rejects_both_sources(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["apriltag_runner.py", "--image", "a.png", "--cam-id", "DEV_1"]
    )
    with pytest.raises(SystemExit):
        ar.parse_args()
