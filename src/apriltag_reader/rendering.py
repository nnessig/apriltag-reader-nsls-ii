"""Drawing helpers and debug image output."""

import cv2


def draw_detections(gray, detections, color=(0, 255, 0)):
    # Draw tag outlines, centers, and IDs on a grayscale image
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for d in detections:
        pts = d.corners.astype(int)

        # Draw the tag border
        for i in range(4):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i + 1) % 4])
            cv2.line(out, pt1, pt2, color, 2)

        # Draw the center point and tag ID label
        center = tuple(d.center.astype(int))
        cv2.circle(out, center, 3, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"ID:{d.tag_id}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_edges_overlay(gray, edges):
    # Overlay detected edges on top of the grayscale image
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out[edges > 0] = (0, 255, 255)
    return out


def write_debug_images(base, gray, edges, overlay, edges_overlay):
    # Save the grayscale image, detection overlay, and edge overlay
    cv2.imwrite(f"{base}_gray.png", gray)
    cv2.imwrite(f"{base}_overlay.png", overlay)
    cv2.imwrite(f"{base}_edges_overlay.png", edges_overlay)

    return [f"{base}_gray.png", f"{base}_overlay.png", f"{base}_edges_overlay.png"]
