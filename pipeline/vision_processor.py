"""
Module A — Vision Processor
Camera frame → trace mask + calibration.

Refactored from extraction_pipeline.py (Steps 1–3) with theme-aware
segmentation for different ECG monitor backgrounds.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.transform import hough_line, hough_line_peaks
import logging
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from config import ThemePreset, get_theme_preset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VisionResult:
    """Output of a single frame processing pass."""
    trace_mask: np.ndarray          # uint8 binary mask (0/255)
    normalized_image: np.ndarray    # BGR image after normalization
    mm_per_px_x: float
    mm_per_px_y: float
    calibration_confidence: float
    latency_ms: float
    lead_rows: List[Tuple[int, int, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_frame(
    frame_bgr: np.ndarray,
    theme: ThemePreset,
    skip_perspective: bool = False,
    expected_leads: int = 0,
) -> VisionResult:
    """
    End-to-end vision processing for a single camera frame.

    Args:
        frame_bgr: Raw BGR image (from camera or file).
        theme: Active theme preset controlling segmentation thresholds.
        skip_perspective: If True, skip perspective correction (for live feed
                          where the user holds the phone steady).
        expected_leads: Number of leads the user expects (0 = auto-detect).

    Returns:
        VisionResult with trace mask and spatial calibration.
    """
    t0 = time.perf_counter()

    # Step 1 — Normalize
    img_norm = normalize_image(frame_bgr, skip_perspective=skip_perspective)

    # Step 2 — Isolate trace using theme-aware segmentation
    trace_mask = isolate_trace(img_norm, theme)

    # Step 3 — Morphological cleanup (first pass with default mm/px)
    clean_mask = clean_trace(trace_mask, mm_per_px=0.1)

    # Step 4 — Grid calibration
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    mm_x, mm_y, confidence = calibrate_grid(gray)

    # Re-clean with calibrated scale if confidence is reasonable
    if confidence > 0.5:
        clean_mask = clean_trace(trace_mask, mm_per_px=mm_y)

    # Step 5 — Detect lead rows from the normalized image
    # (Uses image brightness, not the mask, to avoid grid-line confusion)
    lead_rows = detect_lead_rows(img_norm, expected_leads=expected_leads)

    latency = (time.perf_counter() - t0) * 1000.0

    return VisionResult(
        trace_mask=clean_mask,
        normalized_image=img_norm,
        mm_per_px_x=mm_x,
        mm_per_px_y=mm_y,
        calibration_confidence=confidence,
        latency_ms=latency,
        lead_rows=lead_rows,
    )


# ---------------------------------------------------------------------------
# Step 1 — Image normalization
# ---------------------------------------------------------------------------

def normalize_image(
    img_bgr: np.ndarray,
    skip_perspective: bool = False,
) -> np.ndarray:
    """
    Correct perspective, even out illumination, and compensate for lens
    distortion. Returns a normalized BGR image.
    """
    h, w = img_bgr.shape[:2]

    # 1a — Lens distortion correction (mild barrel correction)
    fx = fy = max(h, w) * 0.9
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.05, 0.01, 0, 0, 0], dtype=np.float32)
    img_undist = cv2.undistort(img_bgr, K, dist_coeffs)

    # 1b — Perspective correction (optional)
    if not skip_perspective:
        img_persp = _correct_perspective(img_undist)
    else:
        img_persp = img_undist

    # 1c — CLAHE on L-channel for illumination normalization
    lab = cv2.cvtColor(img_persp, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _correct_perspective(img: np.ndarray) -> np.ndarray:
    """Detect paper corners and apply homographic warp."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        return img

    pts_src = _order_points(approx.reshape(4, 2).astype(np.float32))
    h, w = img.shape[:2]
    pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img, M, (w, h))


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# ---------------------------------------------------------------------------
# Step 2 — Trace isolation (theme-aware)
# ---------------------------------------------------------------------------

def isolate_trace(img_bgr: np.ndarray, theme: ThemePreset) -> np.ndarray:
    """
    Segment the ECG trace using theme-specific HSV thresholds.
    Falls back to adaptive methods if theme thresholds fail.
    Returns binary mask (uint8, 0 or 255).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Primary: threshold using theme HSV range
    lower = np.array(theme.hsv_lower, dtype=np.uint8)
    upper = np.array(theme.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # If trace is dark on bright background, invert logic
    if not theme.trace_is_bright:
        # For black-on-white: detect dark pixels
        mask = cv2.inRange(hsv, lower, upper)

    coverage = mask.sum() / mask.size
    if coverage < 0.005 or coverage > 0.6:
        # Threshold didn't work well → fall back to LAB adaptive method
        log.info("  HSV threshold coverage %.1f%% out of range; "
                 "falling back to adaptive segmentation", coverage * 100)
        mask = _fallback_segment(img_bgr)

    # Unsharp mask to sharpen trace edges
    mask_f = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_f, (0, 0), 1.0)
    sharp = cv2.addWeighted(mask_f, 1.5, blurred, -0.5, 0)
    mask = np.clip(sharp, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def _fallback_segment(img_bgr: np.ndarray) -> np.ndarray:
    """
    Adaptive fallback segmentation using LAB color space.
    Handles unknown or misidentified themes.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_std = float(np.std(lab[:, :, 1]))
    b_std = float(np.std(lab[:, :, 2]))

    if a_std > 8.0 or b_std > 8.0:
        return _kmeans_segment(lab)
    else:
        return _adaptive_otsu_segment(lab[:, :, 0])


def _kmeans_segment(lab: np.ndarray) -> np.ndarray:
    """k-means (k=3) on LAB pixels; return mask of cluster closest to trace."""
    h, w = lab.shape[:2]
    scale = 0.25
    small = cv2.resize(lab, (int(w * scale), int(h * scale)))
    pixels = small.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # Trace = darkest + most chromatic cluster
    scores = -centers[:, 0] + np.sqrt(centers[:, 1] ** 2 + centers[:, 2] ** 2)
    trace_cluster = int(np.argmax(scores))

    # Full resolution
    pixels_full = lab.reshape(-1, 3).astype(np.float32)
    dists = np.linalg.norm(pixels_full - centers[trace_cluster], axis=1)
    threshold = np.percentile(dists, 30)
    mask = (dists <= threshold).reshape(h, w).astype(np.uint8) * 255
    return mask


def _adaptive_otsu_segment(gray: np.ndarray) -> np.ndarray:
    """Tile-based Otsu thresholding."""
    tile_h, tile_w = gray.shape[0] // 4, gray.shape[1] // 4
    mask = np.zeros_like(gray, dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            tile = gray[y0:y1, x0:x1]
            _, t = cv2.threshold(
                tile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            mask[y0:y1, x0:x1] = t
    return mask


# ---------------------------------------------------------------------------
# Step 3 — Morphological cleanup
# ---------------------------------------------------------------------------

def clean_trace(mask: np.ndarray, mm_per_px: float = 0.1) -> np.ndarray:
    """
    Remove grid residuals, rebuild trace continuity.
    Kernel sizes are dynamic based on calibration (mm_per_px).
    """
    blurred = cv2.GaussianBlur(mask, (0, 0), 1.2)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Erosion — remove thin grid lines
    ek = max(3, round(0.3 / mm_per_px))
    ek = ek if ek % 2 == 1 else ek + 1
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (ek, ek))
    eroded = cv2.erode(binary, erode_k, iterations=1)

    # Horizontal dilation — rebuild signal continuity
    dk = max(7, round(0.7 / mm_per_px))
    dk = dk if dk % 2 == 1 else dk + 1
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, 1))
    dilated = cv2.dilate(eroded, dilate_k, iterations=1)

    # Connected component filtering — keep wide, signal-like blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )
    min_area = 0.005 * mask.size
    clean = np.zeros_like(dilated)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w / (h_comp + 1e-6)
        if area >= min_area and aspect > 2.0:
            clean[labels == i] = 255

    return clean


# ---------------------------------------------------------------------------
# Step 4.5 — Automatic lead row detection
# ---------------------------------------------------------------------------

def detect_lead_rows(
    img_bgr: np.ndarray,
    min_leads: int = 1,
    max_leads: int = 12,
    expected_leads: int = 0,
) -> List[Tuple[int, int, str]]:
    """
    Automatically detect horizontal lead strips in a multi-lead ECG image.

    Instead of analyzing the trace mask (which includes grid lines),
    this examines the original image's brightness profile to find dark
    horizontal separator bands between lead panels.

    Args:
        img_bgr: Normalized BGR image.
        min_leads: Minimum expected lead count.
        max_leads: Maximum expected lead count.
        expected_leads: User-specified lead count (0 = auto-detect).
                        When > 0, constrains detection to return exactly
                        this many leads.

    Returns:
        List of (y_start, y_end, lead_name) tuples.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Compute column-wise variance for each row.
    # Rows with actual waveform traces have high variance (bright trace on
    # dark background), while separator rows are uniformly dark/bright.
    row_brightness = gray.mean(axis=1).astype(np.float32)

    # Also compute row-wise standard deviation — lead rows have high variance
    # (waveform signal), separators are uniform.
    row_std = gray.astype(np.float32).std(axis=1)

    # Combine: trace-active rows are both brighter and have higher variance
    # Normalize each to [0, 1]
    b_max = row_brightness.max() if row_brightness.max() > 0 else 1.0
    s_max = row_std.max() if row_std.max() > 0 else 1.0
    activity = (row_brightness / b_max * 0.5 + row_std / s_max * 0.5)

    # Smooth heavily
    kernel_size = max(5, h // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(
        activity.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    # Skip top/bottom margins (title and axis labels)
    margin = int(h * 0.03)
    analysis = smoothed[margin:h - margin]

    if analysis.max() < 1e-6:
        log.warning("  No activity found for lead detection")
        return [(0, h, "Lead_I")]

    # Adaptive threshold: look for valleys below 40% of median activity
    median_activity = np.median(analysis[analysis > np.percentile(analysis, 20)])
    threshold = median_activity * 0.40

    is_active = analysis > threshold

    # Find contiguous active regions
    regions = []
    in_region = False
    start = 0
    for i in range(len(is_active)):
        if is_active[i] and not in_region:
            start = i
            in_region = True
        elif not is_active[i] and in_region:
            regions.append((start + margin, i + margin))
            in_region = False
    if in_region:
        regions.append((start + margin, h - margin))

    # Merge regions separated by tiny gaps (< 1.5% of image height)
    merge_gap = max(3, int(h * 0.015))
    merged = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    regions = merged

    # Filter out tiny regions (< 8% of image height)
    min_height = max(15, int(h * 0.08))
    regions = [(s, e) for s, e in regions if (e - s) >= min_height]

    if not regions:
        log.warning("  No lead regions found; treating entire image as one lead")
        return [(0, h, "Lead_I")]

    # If user specified a lead count, constrain the result
    if expected_leads > 0:
        if len(regions) == expected_leads:
            pass  # perfect match
        elif len(regions) > expected_leads:
            # Too many regions detected — keep the N largest
            regions.sort(key=lambda r: r[1] - r[0], reverse=True)
            regions = sorted(regions[:expected_leads], key=lambda r: r[0])
            log.info("  User expects %d leads; kept %d largest regions",
                     expected_leads, len(regions))
        else:
            # Too few regions — fall back to even splitting
            log.info("  User expects %d leads but detected %d; "
                     "falling back to even split",
                     expected_leads, len(regions))
            regions = _even_split(h, expected_leads, margin)
    else:
        # Auto mode: just cap at max_leads
        if len(regions) > max_leads:
            regions.sort(key=lambda r: r[1] - r[0], reverse=True)
            regions = sorted(regions[:max_leads], key=lambda r: r[0])

    # Assign lead names
    lead_names_map = [
        "Lead_I", "Lead_II", "Lead_III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    lead_rows = []
    for i, (y_start, y_end) in enumerate(regions):
        name = lead_names_map[i] if i < len(lead_names_map) else f"Lead_{i+1}"
        lead_rows.append((y_start, y_end, name))
        log.info("  Detected %s: rows %d-%d (height=%d)",
                 name, y_start, y_end, y_end - y_start)

    log.info("  Total leads detected: %d", len(lead_rows))
    return lead_rows


def _even_split(
    h: int, n: int, margin: int = 0,
) -> List[Tuple[int, int]]:
    """Evenly split image height into n lead strips with small gaps."""
    usable = h - 2 * margin
    strip_h = usable // n
    gap = max(2, strip_h // 10)  # 10% gap between strips
    regions = []
    for i in range(n):
        y0 = margin + i * strip_h + (gap if i > 0 else 0)
        y1 = margin + (i + 1) * strip_h - (gap if i < n - 1 else 0)
        regions.append((y0, y1))
    return regions


# ---------------------------------------------------------------------------
# Step 5 — Grid calibration
# ---------------------------------------------------------------------------

def calibrate_grid(
    img_gray: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Detect grid lines and compute mm/px ratios.

    Returns:
        (mm_per_px_x, mm_per_px_y, confidence)
    """
    edges = cv2.Canny(img_gray, 30, 80)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)

    try:
        h_space, angles, dists = hough_line(edges, theta=tested_angles)
        _, angles_peak, dists_peak = hough_line_peaks(
            h_space, angles, dists,
            min_distance=5,
            threshold=0.3 * h_space.max(),
        )
    except Exception:
        return 0.1, 0.1, 0.0

    horiz = [d for a, d in zip(angles_peak, dists_peak)
             if abs(a) < np.radians(10)]
    vert = [d for a, d in zip(angles_peak, dists_peak)
            if abs(a - np.pi / 2) < np.radians(10)]

    mm_x = _infer_mm_per_px(vert)
    mm_y = _infer_mm_per_px(horiz)

    ratio = min(mm_x, mm_y) / max(mm_x, mm_y)
    confidence = float(np.clip(ratio, 0, 1))

    return mm_x, mm_y, confidence


def _infer_mm_per_px(positions: list) -> float:
    """Match line spacings to ECG grid periodicities (1mm/5mm)."""
    if len(positions) < 2:
        return 0.1

    positions = sorted(positions)
    spacings = np.diff(positions)
    median_spacing = float(np.median(spacings))

    for grid_mm in [5.0, 1.0]:
        ratio = grid_mm / median_spacing
        if 0.01 <= ratio <= 1.0:
            return ratio

    return 0.1
