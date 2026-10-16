# Vimba AprilTag Reader

`apriltag_runner.py` is a flexible AprilTag detection tool that works with either:

* **Still images**, or
* **An Allied Vision (AVT) camera** using the **Vimba X** SDK.

It supports optional preprocessing, pose estimation, debug image saving, and ROS 2 publishing (including TF frames). The script was built for real experimental use, so most features are designed to make the workflow easy, repeatable, and debuggable.

---

## Features

* Detect AprilTags from:

  * Images (`--image path/to/file.png`)
  * Live camera stream (`--cam-id DEV_XXXXXX`)
* Optional real-time image preprocessing:

  * CLAHE
  * Sharpening
  * Upscaling
  * Canny edge overlay
* Optional 6-DoF pose estimation (requires tag size + camera intrinsics)
* ROS 2 publishing:

  * `/apriltag_ids` (Int32MultiArray)
  * `/apriltag_poses` (PoseArray)
  * TF frames: `camera -> tag_<id>`
* Debug image saving:

  * `<basename>_gray.png`
  * `<basename>_overlay.png`
  * `<basename>_edges_overlay.png`
* Single-frame snapshot mode (for figures, calibration, etc.)

---

## Validation Status

**Calibration:** validated on a representative run. 20 of 24 images
were retained after filtering, with an RMS reprojection error of
1.2886 pixels. Calibration success depends strongly on imaging
conditions (full board visible, sharp focus, even lighting, enough
pixels per feature, low-distortion lens). See
[Camera Calibration](#camera-calibration-charuco).

**Detection:** validated under controlled conditions. Over a 516-frame
evaluation sequence, all visible tags were identified in 515 frames, a
per-frame detection success rate of **99.81%**. This measures whether
every visible tag was detected in a frame. It is not a pose accuracy.
Test conditions were bright, even lighting, manually focused optics,
tuned camera exposure and gain, rigidly mounted tags of appropriate
size, and the `tag36h11` family with an Allied Vision Mako G-131C GigE
camera. Detection was sensitive to lighting, focus, tag size in the
image, viewing angle, and reflections, and performance was worse under
uneven lighting. The single missed frame fits those known failure
modes. Results from other setups may differ.

**Pose output:** working in runtime testing with calibrated
intrinsics. The system produced pose output across continuous camera
streams. Optional temporal filtering is available (`--pose-window`,
`--pose-outlier-thresh`).

**Pose accuracy (6-DoF): not yet measured.** For each detected tag the
tool estimates the full 6-DoF pose: position (X, Y, Z) and orientation
(rotation), relative to the camera. Pose is only computed when
`--tag-size` and the camera intrinsics (`--fx --fy --cx --cy`) are
given. The translational and rotational error of these estimates has
not been verified against ground-truth positions, so no precision
figure is claimed. This testing is planned, for example at LIX once
that stage of implementation is reached.

**Not yet implemented:** transforms into a robot coordinate frame.
Poses are relative to the camera only, and the tool is not used for
closed-loop robot motion.

Until pose accuracy is measured, treat reported poses as unverified.
Also recalibrate if the camera, lens, or focus changes.

---

## Installation

### Python dependencies

```
pip install -r requirements.txt
```

OpenCV is pinned to 4.9 through 4.13 in `requirements.txt`. See the note under
[Camera Calibration](#camera-calibration-charuco) for why.

### Optional (camera mode)

* Allied Vision **Vimba X SDK** installed
* Python module: `vmbpy` (provided with Vimba X)

### Optional (ROS 2 mode)

* ROS 2 (e.g., Humble) with Python bindings:

  * `rclpy`
  * `geometry_msgs`
  * `tf2_ros`

---

## Usage

### Run on an Image

```
python3 apriltag_runner.py \
  --image tag_images/test_tag.png \
  --save-debug \
  --tag-size 0.025 \
  --fx 1200 --fy 1200 --cx 640 --cy 512
```

This will:

* Detect tags in the image
* Print detections to the terminal
* Save:

  * `tag_image_gray.png`
  * `tag_image_overlay.png`
  * `tag_image_edges_overlay.png`

---

### Run on an AVT Camera

Basic live detection with preview:

```
python3 apriltag_runner.py \
  --cam-id DEV_000F315BCAB1 \
  --preview
```

Grab a single frame and exit:

```
python3 apriltag_runner.py \
  --cam-id DEV_000F315BCAB1 \
  --single --save-debug --out frame_test
```

This will save debug images with the base name `frame_test`.

---

## ROS 2 Mode

Enable ROS 2 output with `--ros`.

Example:

```
python3 apriltag_runner.py \
  --cam-id DEV_000F315BCAB1 \
  --ros \
  --tag-size 0.025 \
  --fx 1200 --fy 1200 --cx 640 --cy 512 \
  --ros-rate 20 \
  --frame-id camera
```

This will publish:

* `/apriltag_ids` – `Int32MultiArray` of detected tag IDs
* `/apriltag_poses` – `PoseArray` of tag poses in the camera frame
* TF frames – one per tag: `camera -> tag_<id>`

---

---

# Camera Calibration (Charuco)


⚠️ **Always measure your printed Charuco board before calibrating.**
Incorrect print scaling (e.g., "fit to page") is one of the most common causes of calibration failure.


Before using pose estimation with apriltag_runner.py, you should
calibrate the camera to obtain accurate intrinsic parameters (fx, fy,
cx, cy).

A helper script camera_calibration.py is included for this purpose. It
performs Charuco-based camera calibration using images of a Charuco
board.

---

Requirements

```
pip install -r requirements.txt
```

The opencv-contrib-python package is required for the cv2.aruco module.

**OpenCV version matters.** Use OpenCV 4.9 through 4.13 (pinned in
`requirements.txt`). On the same 24 demo images, 4.9 to 4.13 give an RMS
reprojection error of about 1.29 px, while 4.14 gives about 5.46 px and
different intrinsics. OpenCV 5.0 removed the Charuco calibration function this
script uses, so it fails there. The example output below was produced with
OpenCV 4.9 to 4.13. The cause of the 4.14 difference has not been identified.

---

Calibration Board

This script assumes the following Charuco board configuration:

* Squares: 5 × 3
* Square size: 5 mm
* Marker size: 3.5 mm
* Dictionary: DICT_4X4_50

Make sure your printed board matches these parameters or modify the file accordingly.

---

Preparing Calibration Images

1. Place the calibration images in the `calibration_images/` folder.

2. Supported formats:

   *.png
   *.jpg
   *.jpeg

3. Capture ~25–40 images of the Charuco board with:

   * Different orientations
   * Different positions in the frame
   * The board covering different parts of the image
   * Good lighting and sharp focus

The script automatically scans the selected folder and loads all images.

---

Running Calibration

Example:

```
python3 camera_calibration.py --images-dir calibration_images --out output/camera_charuco_calib.yaml
```

Optional flags:

```
--images-dir calibration_images      # folder containing calibration images
--out output/camera_charuco_calib.yaml   # output file name
--show                               # visualize detections during calibration
--min-markers 2                      # minimum detected ArUco markers per frame
```

Example with visualization:

```
python3 camera_calibration.py --images-dir calibration_images --show --out output/camera_charuco_calib.yaml
```

---

Example Output

```
=== Calibration Result ===
Kept frames: 20 / 24
RMS reprojection error: 1.2886 pixels

K (camera matrix):
[[290.13578038   0.         672.32394849]
 [  0.         283.41101199 653.22399757]
 [  0.           0.           1.        ]]

D (dist coeffs):
[ 0.0120927  -0.00889392 -0.0083376  -0.00296499  0.00127125]
```

The script will also print the parameters needed for apriltag_runner.py:

```
Use these in AprilTag runner:

--fx 290.135780
--fy 283.411012
--cx 672.323948
--cy 653.223998
```

Example usage with the AprilTag runner:

```
python3 apriltag_runner.py \
  --cam-id DEV_000F315BCAB1 \
  --tag-size 0.025 \
  --fx 290.135780 \
  --fy 283.411012 \
  --cx 672.323948 \
  --cy 653.223998
```

---

Output File

The calibration script saves a YAML file containing:

```
output/camera_charuco_calib.yaml
```

This file includes:

* Image width and height
* Camera matrix
* Distortion coefficients
* RMS reprojection error

This file can be archived for reproducibility or reused in other vision
pipelines.


---

## Common Calibration Failures & Fixes

The following are common real-world issues encountered when running the `camera_calibration.py` pipeline, along with their causes and fixes.

### Issue: "0 good frames kept"

**Symptoms:**
- ArUco markers are detected (visible with `--show`)
- Calibration still fails with zero usable frames

**Most likely cause:**
- Checkerboard squares are too small to resolve clearly

**Fix:**
- Use a **larger Charuco board** (e.g., 20–30 mm squares instead of 5 mm)

---

## Important Practical Tips

### 1. Verify Printed Board Dimensions

**Do NOT assume your printed board matches the intended size.**

Printer scaling (e.g., “fit to page”) can distort dimensions.

**Always:**
- Use a ruler or calipers
- Measure:
  - Checker square size
  - ArUco marker size

If they do NOT match your expected values:
- Update the parameters in the calibration script accordingly

---

### 2. Avoid Distorting Lenses

Camera lens distortion can significantly impact calibration quality.

Lenses with strong distortion (e.g., wide-angle or ultra-wide) can:
- Warp straight lines
- Make corner detection less reliable
- Increase calibration error

**Recommendation:**
- Use a standard (non-wide) lens
- Avoid wide-angle or ultra-wide lenses when possible

---

### 3. Keep the Board Flat

Printed paper boards can warp slightly.

Even small curvature:
- Breaks corner detection
- Reduces calibration accuracy

**Recommendation:**
- Mount the board on a **rigid flat surface**
  - (cardboard, foam board, etc.)

---

### 4. Image Quality Matters

Ensure:
- Good lighting
- Sharp focus
- Board fills a reasonable portion of the frame
- 25–40 images with varied angles

---

### 5. Debugging Tip

Use:

```
--show
```

This helps confirm:
- Markers are detected
- Whether corner detection is failing

---

## Quick Troubleshooting Checklist

If calibration fails:

1. Increase board size (most common fix)
2. Measure actual printed dimensions
3. Switch off ultra-wide lens
4. Ensure board is flat
5. Verify image quality

---

## AI Disclosure

AI tools were used in developing this project. See [AI_DISCLOSURE.md](AI_DISCLOSURE.md) for details.

---

## License

This project is released under the BSD 3-Clause License. See [LICENSE](LICENSE). The
U.S. Government license terms and disclaimer are in [LICENSE_README](LICENSE_README).
