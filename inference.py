from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

_MODEL: MoGeModel | None = None
_DEVICE: torch.device | None = None


def _resolve_device() -> torch.device:
    preferred = os.getenv("MOGE_DEVICE", "cuda").lower()
    if preferred == "cuda" and not torch.cuda.is_available():
        preferred = "cpu"
    return torch.device(preferred)


def _get_model() -> tuple[MoGeModel, torch.device]:
    global _MODEL
    global _DEVICE

    if _MODEL is not None and _DEVICE is not None:
        return _MODEL, _DEVICE

    _DEVICE = _resolve_device()
    model_id = os.getenv("MOGE_MODEL_ID", "Ruicheng/moge-2-vitl-normal")
    _MODEL = MoGeModel.from_pretrained(model_id).to(_DEVICE)
    _MODEL.eval()
    return _MODEL, _DEVICE


def warm_model() -> None:
    _get_model()


def _decode_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid image data.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def _estimate_height(
    points: np.ndarray, mask: np.ndarray, normal: np.ndarray | None
) -> tuple[float, float, float] | None:
    valid_mask = mask > 0
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        return None

    y_coords = valid_points[:, 1]
    floor_y = float(np.percentile(y_coords, 95))
    ceiling_y = float(np.percentile(y_coords, 5))

    if normal is not None:
        normal_y = normal[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        ceiling_mask = (normal_y > 0.7) & valid_mask
        if floor_mask.any():
            floor_y = float(np.mean(points[floor_mask][:, 1]))
        if ceiling_mask.any():
            ceiling_y = float(np.mean(points[ceiling_mask][:, 1]))

    height = abs(floor_y - ceiling_y)
    return height, floor_y, ceiling_y


def _floor_points(points: np.ndarray, mask: np.ndarray, normal: np.ndarray | None) -> np.ndarray | None:
    valid_mask = mask > 0
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        return None

    if normal is not None:
        normal_y = normal[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        if floor_mask.any():
            return points[floor_mask]

    y_coords = valid_points[:, 1]
    threshold = np.percentile(y_coords, 80)
    floor_points = valid_points[valid_points[:, 1] > threshold]
    return floor_points if floor_points.size else valid_points


def _order_corners(corners: np.ndarray) -> list[list[int]]:
    sums = corners[:, 0] + corners[:, 1]
    diffs = corners[:, 0] - corners[:, 1]
    top_left = corners[int(np.argmin(sums))]
    bottom_right = corners[int(np.argmax(sums))]
    top_right = corners[int(np.argmin(diffs))]
    bottom_left = corners[int(np.argmax(diffs))]
    ordered = [top_left, top_right, bottom_left, bottom_right]
    return [[int(point[0]), int(point[1])] for point in ordered]


def _estimate_corners(contour: np.ndarray) -> list[list[int]]:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    corners = None
    for epsilon_factor in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) >= 4:
            corners = approx.reshape(-1, 2)
        if len(approx) == 4:
            break

    if corners is None:
        corners = hull.reshape(-1, 2)

    if len(corners) > 4:
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        indices = list(
            {
                int(np.argmin(x_coords)),
                int(np.argmax(x_coords)),
                int(np.argmin(y_coords)),
                int(np.argmax(y_coords)),
            }
        )
        if len(indices) >= 4:
            corners = corners[indices[:4]]
        else:
            corners = corners[:4]

    if len(corners) < 4:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)

    return _order_corners(np.array(corners, dtype=np.int32))


def _build_ceiling_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    normal_map: np.ndarray | None,
    points_map: np.ndarray,
    dimensions: dict[str, float],
) -> dict[str, Any]:
    if normal_map is None:
        return {}

    valid_mask = mask > 0
    normal_y = normal_map[:, :, 1]
    ceiling_mask = (normal_y > 0.7) & valid_mask
    if not ceiling_mask.any():
        return {}

    ceiling_uint8 = (ceiling_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(ceiling_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    largest = max(contours, key=cv2.contourArea)
    corners = _estimate_corners(largest)

    overlay = image_bgr.copy()
    cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 3)

    contour_points = largest[:, 0, :]
    corner_indices = []
    snapped_corners: list[list[int]] = []
    for corner in corners:
        deltas = contour_points - np.array(corner)
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        closest_idx = int(np.argmin(distances))
        corner_indices.append(closest_idx)
        snapped = contour_points[closest_idx]
        snapped_corners.append([int(snapped[0]), int(snapped[1])])

    corners = snapped_corners
    sorted_corner_data = sorted(zip(corner_indices, corners))

    for corner in corners:
        cv2.circle(overlay, (corner[0], corner[1]), 8, (255, 0, 255), -1)
    for i in range(len(sorted_corner_data)):
        idx1, _ = sorted_corner_data[i]
        idx2, _ = sorted_corner_data[(i + 1) % len(sorted_corner_data)]
        if idx1 < idx2:
            segment_indices = list(range(idx1, idx2 + 1))
        else:
            segment_indices = list(range(idx1, len(contour_points))) + list(range(0, idx2 + 1))

        total_dist = 0.0
        for j in range(len(segment_indices) - 1):
            pt_a = contour_points[segment_indices[j]]
            pt_b = contour_points[segment_indices[j + 1]]
            if mask[pt_a[1], pt_a[0]] > 0 and mask[pt_b[1], pt_b[0]] > 0:
                pt_a_3d = points_map[pt_a[1], pt_a[0]]
                pt_b_3d = points_map[pt_b[1], pt_b[0]]
                total_dist += float(np.linalg.norm(pt_a_3d - pt_b_3d))

        if total_dist > 0:
            mid_idx = segment_indices[len(segment_indices) // 2]
            mid_pt = contour_points[mid_idx]
            cv2.putText(
                overlay,
                f"{total_dist:.2f}m",
                (int(mid_pt[0]) - 40, int(mid_pt[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
            )

    pts = largest[:, 0, :]
    min_y = int(np.min(pts[:, 1]))
    min_x = int(np.min(pts[:, 0]))
    top_pts = pts[pts[:, 1] < min_y + 30]
    left_pts = pts[pts[:, 0] < min_x + 30]

    if len(top_pts) > 1:
        top_pts_sorted = top_pts[np.argsort(top_pts[:, 0])]
        mid_x = int((top_pts_sorted[0][0] + top_pts_sorted[-1][0]) // 2)
        mid_y = int((top_pts_sorted[0][1] + top_pts_sorted[-1][1]) // 2)
        cv2.putText(
            overlay,
            f"{dimensions['depth']:.2f}m",
            (mid_x - 60, mid_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 255),
            3,
        )

    if len(left_pts) > 1:
        left_pts_sorted = left_pts[np.argsort(left_pts[:, 1])]
        mid_x = int((left_pts_sorted[0][0] + left_pts_sorted[-1][0]) // 2)
        mid_y = int((left_pts_sorted[0][1] + left_pts_sorted[-1][1]) // 2)
        cv2.putText(
            overlay,
            f"{dimensions['width']:.2f}m",
            (mid_x + 20, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 255),
            3,
        )

    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        return {}

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "ceiling_overlay_base64": f"data:image/png;base64,{payload}",
        "ceiling_corners": corners,
    }


def infer_dimensions(
    image_bytes: bytes, calibration_height_m: float | None
) -> dict[str, Any]:
    model, device = _get_model()
    image_bgr, image_rgb = _decode_image(image_bytes)
    tensor = torch.from_numpy(image_rgb).to(device).float() / 255.0
    tensor = tensor.permute(2, 0, 1)

    with torch.no_grad():
        output = model.infer(tensor)

    points = output["points"].detach().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()
    normal = output.get("normal")
    normal_map = normal.detach().cpu().numpy() if normal is not None else None

    height_data = _estimate_height(points, mask, normal_map)
    if height_data is None:
        raise RuntimeError("Unable to estimate room height.")
    height, _, _ = height_data

    floor_points = _floor_points(points, mask, normal_map)
    if floor_points is None:
        raise RuntimeError("Unable to determine floor points.")

    x_coords = floor_points[:, 0]
    z_coords = floor_points[:, 2]
    width = float(np.max(x_coords) - np.min(x_coords))
    depth = float(np.max(z_coords) - np.min(z_coords))
    area = width * depth

    calibration_factor = None
    points_for_overlay = points
    if calibration_height_m and height > 0:
        calibration_factor = float(calibration_height_m / height)
        width *= calibration_factor
        depth *= calibration_factor
        height *= calibration_factor
        area *= calibration_factor ** 2
        points_for_overlay = points * calibration_factor

    dimensions = {
        "width": width,
        "depth": depth,
        "height": height,
        "area": area,
        "unit": "m",
        "calibration_factor": calibration_factor,
    }
    overlay = _build_ceiling_overlay(image_bgr, mask, normal_map, points_for_overlay, dimensions)
    return {"dimensions": dimensions, **overlay}
