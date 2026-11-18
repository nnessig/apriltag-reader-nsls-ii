"""AprilTag detection core."""

import sys

import numpy as np

from .pose import rotation_matrix_to_rpy

# Try importing the AprilTag detector implementation
# pupil_apriltags is used here for speed and stability
try:
    from pupil_apriltags import Detector
except ImportError:
    print(
        "ERROR: pupil_apriltags not found. Install via: pip install pupil-apriltags",
        file=sys.stderr,
    )
    raise

# Cache detector instances so the detector does not need to be recreated
# every frame for the same settings
_DETECTOR_CACHE = {}


class _WrappedDetection:
    def __init__(self, det):
        self.__dict__.update(det.__dict__)

    @property
    def pose_rpy(self):
        return np.array(rotation_matrix_to_rpy(self.pose_R))


def detect_tags(
    gray,
    families="tag36h11",
    nthreads=1,
    decimate=1.0,
    sigma=0.0,
    refine_edges=False,
    decode_sharpening=0.25,
    do_pose=False,
    tag_size=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    **kwargs,
):
    # The CLI documents comma-separated families, but the detector wants them space-separated
    families = " ".join(families.replace(",", " ").split())

    # Reuse a detector if one with the same settings already exists
    key = (families, nthreads, decimate, sigma, bool(refine_edges), decode_sharpening)
    detector = _DETECTOR_CACHE.get(key)

    if detector is None:
        detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=decimate,
            quad_sigma=sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=False,
        )
        _DETECTOR_CACHE[key] = detector

    # Run detection with or without pose estimation
    if do_pose:
        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size,
        )
    else:
        results = detector.detect(
            gray,
            estimate_tag_pose=False,
        )

    return [_WrappedDetection(d) for d in results]
