# AprilTag Runner

`apriltag_runner.py` is a flexible AprilTag detection tool that works with either:

- **Still images**, or  
- **An Allied Vision (AVT) camera** using the **Vimba X** SDK.

It supports optional preprocessing, pose estimation, debug image saving, and ROS 2 publishing (including TF frames). The script was built for real experimental use, so most features are designed to make the workflow easy, repeatable, and debuggable.

---

## Features

- Detect AprilTags from:
  - Images (`--image path/to/file.png`)
  - Live camera stream (`--cam-id DEV_XXXXXX`)
- Optional real-time image preprocessing:
  - CLAHE
  - Sharpening
  - Upscaling
  - Canny edge overlay
- Optional 6-DoF pose estimation (requires tag size + camera intrinsics)
- ROS 2 publishing:
  - `/apriltag_ids` (Int32MultiArray)
  - `/apriltag_poses` (PoseArray)
  - TF frames: `camera -> tag_<id>`
- Debug image saving:
  - `<basename>_gray.png`
  - `<basename>_overlay.png`
  - `<basename>_edges_overlay.png`
- Single-frame snapshot mode (for figures, calibration, etc.)

---

## Installation

### Python dependencies

```
pip install pupil-apriltags opencv-python numpy
```

### Optional (camera mode)

- Allied Vision **Vimba X SDK** installed
- Python module: `vmbpy` (provided with Vimba X)

### Optional (ROS 2 mode)

- ROS 2 (e.g., Humble) with Python bindings:
  - `rclpy`
  - `geometry_msgs`
  - `tf2_ros`

---

## Usage

### Run on an Image

```
python3 apriltag_runner.py \
  --image path/to/tag_image.png \
  --save-debug \
  --tag-size 0.025 \
  --fx 1200 --fy 1200 --cx 640 --cy 512
```

This will:

- Detect tags in the image
- Print detections to the terminal
- Save:
  - `tag_image_gray.png`
  - `tag_image_overlay.png`
  - `tag_image_edges_overlay.png`

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

- `/apriltag_ids` – `Int32MultiArray` of detected tag IDs  
- `/apriltag_poses` – `PoseArray` of tag poses in the camera frame  
- TF frames – one per tag: `camera -> tag_<id>`


