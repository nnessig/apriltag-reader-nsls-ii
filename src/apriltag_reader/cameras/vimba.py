"""Allied Vision (Vimba X) camera streaming. ``vmbpy`` is optional."""

import queue
import sys
from contextlib import contextmanager

# Try importing Vimba X support for Allied Vision camera streaming
_vimba_available = False
try:
    import vmbpy as vmb

    _vimba_available = True
except Exception:
    pass


@contextmanager
def open_stream(cam_id, exposure_us=None, gain=None):
    """Open the camera ``cam_id`` and start streaming.

    Yields a small queue holding the newest complete frames as numpy arrays.
    Streaming is stopped and the camera closed when the context exits.
    """
    # Camera mode requires Vimba X support
    if not _vimba_available:
        print("ERROR: vmbpy (Vimba X) not available.", file=sys.stderr)
        sys.exit(1)

    # Small queue to hold the latest frames from the camera callback
    q = queue.Queue(maxsize=2)

    def handler(cam, stream, frame):
        # Camera callback: receive frame, convert to numpy, and keep only the newest frames
        try:
            if frame.get_status() == vmb.FrameStatus.Complete:
                img = frame.as_numpy_ndarray().copy()

                if q.full():
                    try:
                        _ = q.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    q.put_nowait(img)
                except queue.Full:
                    pass
        finally:
            # Re-queue the frame buffer back to the stream
            stream.queue_frame(frame)

    # Open Vimba system and search for the requested camera
    with vmb.VmbSystem.get_instance() as vmbs:
        cams = vmbs.get_all_cameras()
        cam = next((c for c in cams if c.get_id() == cam_id), None)

        if cam is None:
            print(
                f"ERROR: Camera {cam_id} not found. Available: {[c.get_id() for c in cams]}",
                file=sys.stderr,
            )
            sys.exit(1)

        with cam:
            # Try forcing Mono8 pixel format if available
            try:
                cam.set_pixel_format(vmb.PixelFormat.Mono8)
            except Exception:
                pass

            # Optionally set exposure
            if exposure_us is not None:
                try:
                    cam.ExposureTime.set(float(exposure_us))
                except Exception:
                    pass

            # Optionally set gain
            if gain is not None:
                try:
                    cam.Gain.set(float(gain))
                except Exception:
                    pass

            # Start streaming frames into the callback
            cam.start_streaming(handler=handler, buffer_count=8)
            try:
                yield q
            finally:
                cam.stop_streaming()
