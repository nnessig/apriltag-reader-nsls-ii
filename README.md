# AprilTag Runner

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

## Installation

### Python dependencies

```
pip install pupil-apriltags opencv-python numpy
```

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
pip install opencv-python opencv-contrib-python numpy
```

The opencv-contrib-python package is required for the cv2.aruco module.

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

### 2. Avoid Ultra-Wide Lenses

Ultra-wide lenses introduce **heavy distortion**, which:
- Makes small boards harder to detect
- Reduces calibration accuracy

**Recommendation:**
- Use the **main (standard) camera lens**
- Avoid wide-angle or ultra-wide modes

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
