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

    # Step 4 — Grid calibration (Hough-based, then spectral fallback)
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    mm_x, mm_y, confidence = calibrate_grid(gray)

    # Spectral fallback (ECG-Image-Kit technique) when Hough confidence is low
    if confidence < 0.4:
        try:
            from pipeline.signal_processor import spectral_grid_estimation
            grid_h_px, grid_v_px = spectral_grid_estimation(gray)
            if grid_h_px > 2 and grid_v_px > 2:
                # Standard ECG: small grid = 1mm, large grid = 5mm
                # Choose the most likely match
                for grid_mm in [1.0, 5.0]:
                    ratio_x = grid_mm / grid_v_px  # vertical image lines = horizontal spacing
                    ratio_y = grid_mm / grid_h_px  # horizontal image lines = vertical spacing
                    if 0.01 <= ratio_x <= 1.0 and 0.01 <= ratio_y <= 1.0:
                        mm_x, mm_y = ratio_x, ratio_y
                        confidence = max(confidence, 0.6)
                        log.info("  Spectral grid calibration: mm/px = (%.4f, %.4f)",
                                mm_x, mm_y)
                        break
        except Exception as e:
            log.warning("  Spectral grid calibration failed: %s", e)

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
    Automatically detect lead strips in single or multi-lead ECG images.

    Handles both:
      - Simple row-band layouts (1-3 leads stacked vertically)
      - 2D grid layouts (12-lead monitor: 2 cols × 6 rows or 4 cols × 3 rows)

    Args:
        img_bgr: Normalized BGR image.
        min_leads: Minimum expected lead count.
        max_leads: Maximum expected lead count.
        expected_leads: User-specified lead count (0 = auto-detect).

    Returns:
        List of (y_start, y_end, lead_name) tuples.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # First, try to detect a 2D grid layout (12-lead monitor)
    grid_result = _detect_grid_layout(img_bgr, expected_leads)
    if grid_result is not None and len(grid_result) >= 4:
        log.info("  Detected grid layout with %d leads", len(grid_result))
        return grid_result

    # Fall back to row-band detection for simpler layouts
    return _detect_row_bands(img_bgr, min_leads, max_leads, expected_leads)


def _detect_grid_layout(
    img_bgr: np.ndarray,
    expected_leads: int = 0,
) -> Optional[List[Tuple[int, int, str]]]:
    """
    Detect a 2D grid of ECG lead cells (e.g., 12-lead monitor display).

    Strategy:
      1. Use green-channel HSV mask to find waveform pixels
      2. Morphological closing to connect nearby waveform clusters
      3. Find the largest connected component (the waveform grid area)
      4. Within that area, split into columns and rows using density valleys
      5. Assign standard lead names based on grid position
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Detect green waveform pixels
    green_mask = cv2.inRange(hsv,
                             np.array([30, 40, 40], dtype=np.uint8),
                             np.array([90, 255, 255], dtype=np.uint8))

    # Morphological closing with large kernel to bridge gaps between lead cells
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find connected components — the largest one is the waveform grid
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    if n_labels < 2:
        return None

    # Skip label 0 (background), find largest component
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = np.argmax(areas) + 1
    main_x = stats[main_label, cv2.CC_STAT_LEFT]
    main_y = stats[main_label, cv2.CC_STAT_TOP]
    main_w = stats[main_label, cv2.CC_STAT_WIDTH]
    main_h = stats[main_label, cv2.CC_STAT_HEIGHT]

    # The waveform grid should be a substantial portion of the image
    if main_w < w * 0.25 or main_h < h * 0.25:
        return None

    x_min, x_max = main_x, main_x + main_w
    y_min, y_max = main_y, main_y + main_h

    log.info("  Grid ROI: x=[%d,%d], y=[%d,%d] (%dx%d)",
             x_min, x_max, y_min, y_max, main_w, main_h)

    # Analyze green density within the ROI
    green_roi = green_mask[y_min:y_max, x_min:x_max]

    from scipy.ndimage import gaussian_filter1d

    # Column density profile — find vertical dividers
    col_density = green_roi.astype(np.float32).mean(axis=0) / 255.0
    col_smooth = gaussian_filter1d(col_density, sigma=max(3, main_w // 60))

    # Row density profile — find horizontal dividers
    row_density = green_roi.astype(np.float32).mean(axis=1) / 255.0
    row_smooth = gaussian_filter1d(row_density, sigma=max(3, main_h // 40))

    # Find column splits
    col_med = np.median(col_smooth[col_smooth > 0.005]) if (col_smooth > 0.005).any() else 0.1
    col_valleys = col_smooth < col_med * 0.2
    col_splits = _find_splits(col_valleys, min_size=main_w // 20)

    if len(col_splits) < 2:
        return None

    n_cols = len(col_splits)
    log.info("  Grid: detected %d columns", n_cols)

    # For each column, find row splits
    all_cells = []
    for ci, (cx_start, cx_end) in enumerate(col_splits):
        col_green = green_mask[y_min:y_max, x_min + cx_start:x_min + cx_end]
        row_dens = col_green.astype(np.float32).mean(axis=1) / 255.0
        row_sm = gaussian_filter1d(row_dens, sigma=max(3, main_h // 40))

        row_med = np.median(row_sm[row_sm > 0.005]) if (row_sm > 0.005).any() else 0.1
        row_vals = row_sm < row_med * 0.15
        row_splits = _find_splits(row_vals, min_size=main_h // 20)

        if not row_splits:
            row_splits = [(0, main_h)]

        for ri, (ry_start, ry_end) in enumerate(row_splits):
            abs_y_start = y_min + ry_start
            abs_y_end = y_min + ry_end
            all_cells.append((abs_y_start, abs_y_end, x_min + cx_start,
                              x_min + cx_end, ci, ri))

    if len(all_cells) < 6:
        return None

    # Sort: by column then by row
    all_cells.sort(key=lambda c: (c[4], c[5]))

    # Standard 12-lead naming by grid position
    if n_cols == 2:
        name_grid = [
            ["Lead_I", "Lead_II", "Lead_III", "aVR", "aVL", "aVF"],
            ["V1", "V2", "V3", "V4", "V5", "V6"],
        ]
    elif n_cols == 4:
        name_grid = [
            ["Lead_I", "Lead_II", "Lead_III"],
            ["aVR", "aVL", "aVF"],
            ["V1", "V2", "V3"],
            ["V4", "V5", "V6"],
        ]
    elif n_cols == 3:
        name_grid = [
            ["Lead_I", "Lead_II", "Lead_III", "aVR"],
            ["aVL", "aVF", "V1", "V2"],
            ["V3", "V4", "V5", "V6"],
        ]
    else:
        name_grid = None

    results = []
    for cell in all_cells:
        y_start, y_end, x_start, x_end, ci, ri = cell
        if name_grid and ci < len(name_grid) and ri < len(name_grid[ci]):
            name = name_grid[ci][ri]
        else:
            idx = len(results)
            lead_names = ["Lead_I", "Lead_II", "Lead_III",
                          "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"]
            name = lead_names[idx] if idx < len(lead_names) else f"Lead_{idx+1}"

        results.append((y_start, y_end, name))
        log.info("  Grid cell %s: rows %d-%d (col %d, row %d)",
                 name, y_start, y_end, ci, ri)

    return results


def _find_splits(valleys: np.ndarray, min_size: int = 10) -> List[Tuple[int, int]]:
    """Find contiguous non-valley regions (segments) with minimum size."""
    segments = []
    in_seg = False
    start = 0
    n = len(valleys)

    for i in range(n):
        if not valleys[i] and not in_seg:
            start = i
            in_seg = True
        elif valleys[i] and in_seg:
            if i - start >= min_size:
                segments.append((start, i))
            in_seg = False
    if in_seg and n - start >= min_size:
        segments.append((start, n))

    return segments


def _detect_row_bands(
    img_bgr: np.ndarray,
    min_leads: int = 1,
    max_leads: int = 12,
    expected_leads: int = 0,
) -> List[Tuple[int, int, str]]:
    """
    Detect horizontal lead strips for simple stacked layouts.
    Original row-band detection approach.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    row_brightness = gray.mean(axis=1).astype(np.float32)
    row_std = gray.astype(np.float32).std(axis=1)

    b_max = row_brightness.max() if row_brightness.max() > 0 else 1.0
    s_max = row_std.max() if row_std.max() > 0 else 1.0
    activity = (row_brightness / b_max * 0.5 + row_std / s_max * 0.5)

    kernel_size = max(5, h // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(
        activity.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    margin = int(h * 0.03)
    analysis = smoothed[margin:h - margin]

    if analysis.max() < 1e-6:
        log.warning("  No activity found for lead detection")
        return [(0, h, "Lead_I")]

    median_activity = np.median(analysis[analysis > np.percentile(analysis, 20)])
    threshold = median_activity * 0.40

    is_active = analysis > threshold

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

    merge_gap = max(3, int(h * 0.015))
    merged = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    regions = merged

    min_height = max(15, int(h * 0.08))
    regions = [(s, e) for s, e in regions if (e - s) >= min_height]

    if not regions:
        log.warning("  No lead regions found; treating entire image as one lead")
        return [(0, h, "Lead_I")]

    if expected_leads > 0:
        if len(regions) == expected_leads:
            pass
        elif len(regions) > expected_leads:
            regions.sort(key=lambda r: r[1] - r[0], reverse=True)
            regions = sorted(regions[:expected_leads], key=lambda r: r[0])
        else:
            log.info("  User expects %d leads but detected %d; "
                     "falling back to even split",
                     expected_leads, len(regions))
            regions = _even_split(h, expected_leads, margin)
    else:
        if len(regions) > max_leads:
            regions.sort(key=lambda r: r[1] - r[0], reverse=True)
            regions = sorted(regions[:max_leads], key=lambda r: r[0])

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
    gap = max(2, strip_h // 10)
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
