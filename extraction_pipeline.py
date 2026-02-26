"""
ECG Signal Extraction Pipeline v2
Supports: paper printouts, digital monitor screenshots, software plot exports, video frames.

Dependencies:
    pip install opencv-python numpy scipy scikit-image matplotlib
"""

import cv2
import numpy as np
from scipy import signal, ndimage
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class SourceType(Enum):
    PAPER      = auto()   # Physical printout photographed by phone
    SCREENSHOT = auto()   # Digital monitor or software screenshot
    VIDEO      = auto()   # Phone video of a live monitor


@dataclass
class UserScale:
    """Required for SCREENSHOT and VIDEO sources; ignored for PAPER."""
    mv_per_div: float       # e.g. 0.5 for a 0.5mV/div display
    seconds_per_div: float  # e.g. 0.2 for a 200ms/div display


@dataclass
class CalibrationResult:
    px_per_div_x: float
    px_per_div_y: float
    mv_per_px: float
    sec_per_px: float
    source_type: SourceType
    confidence: float           # 0–1; lower for paper with few detected lines


@dataclass
class LeadSignal:
    name: str
    voltage_mv: np.ndarray
    time_s: np.ndarray
    snr_db: float
    continuity_pct: float
    baseline_drift_mv: np.ndarray


@dataclass
class ExtractionResult:
    leads: list[LeadSignal]
    calibration: CalibrationResult
    passed_qc: bool
    source_type: SourceType
    rejection_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — Source detection & normalization
# ---------------------------------------------------------------------------

def detect_source_type(img_bgr: np.ndarray) -> SourceType:
    """
    Classify input as PAPER, SCREENSHOT, or VIDEO.
    Heuristics:
      - Mean brightness < 80 → likely dark-background screen source
      - High background uniformity (low std in large central region) → SCREENSHOT
      - Otherwise → PAPER
    Video frames are distinguished externally via the `is_video_frame` flag in
    normalize_image(); single images default to SCREENSHOT for dark inputs.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    bg_std = float(np.std(gray[gray < 50]))   # variation among dark pixels

    if mean_brightness < 80:
        log.info("  Dark background detected → SCREENSHOT source")
        return SourceType.SCREENSHOT
    else:
        log.info("  Light background detected → PAPER source")
        return SourceType.PAPER


def normalize_image(
    img_bgr: np.ndarray,
    is_video_frame: bool = False,
    prev_frames: Optional[list[np.ndarray]] = None,
) -> tuple[np.ndarray, SourceType]:
    """
    Route image through the correct normalization path based on source type.
    Returns (normalized_image, source_type).
    """
    log.info("Step 1: Normalizing image")

    source = SourceType.VIDEO if is_video_frame else detect_source_type(img_bgr)

    if source == SourceType.PAPER:
        img_out = _normalize_paper(img_bgr)

    elif source == SourceType.SCREENSHOT:
        img_out = _normalize_screenshot(img_bgr)

    elif source == SourceType.VIDEO:
        img_out = _normalize_video(img_bgr, prev_frames or [])

    log.info("  Normalized as %s: %dx%d", source.name, img_out.shape[1], img_out.shape[0])
    return img_out, source


def _normalize_paper(img: np.ndarray) -> np.ndarray:
    """Perspective warp + lens undistort + CLAHE illumination correction."""
    h, w = img.shape[:2]
    # Approximate phone lens distortion correction
    fx = fy = max(h, w) * 0.9
    K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)
    dist = np.array([-0.05, 0.01, 0, 0, 0], dtype=np.float32)
    img = cv2.undistort(img, K, dist)
    img = _correct_perspective(img)

    # CLAHE on L-channel to even out illumination across the page
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _normalize_screenshot(img: np.ndarray) -> np.ndarray:
    """Crop UI chrome; no perspective correction needed for screenshots."""
    return _crop_waveform_region(img)


def _normalize_video(img: np.ndarray, prev_frames: list[np.ndarray]) -> np.ndarray:
    """
    Temporal averaging over up to 5 frames to suppress monitor flicker,
    then crop to waveform region.
    """
    frames = (prev_frames[-4:] + [img]) if prev_frames else [img]
    averaged = np.mean([f.astype(np.float32) for f in frames], axis=0).astype(np.uint8)
    return _crop_waveform_region(averaged)


def _crop_waveform_region(img: np.ndarray) -> np.ndarray:
    """
    Auto-detect and crop to the waveform panel(s), discarding UI chrome.
    Strategy: find the largest contiguous dark region (background of ECG display)
    that excludes bright UI elements (buttons, headers, sidebars).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold: keep only near-black pixels (ECG display background)
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Find the largest dark rectangle — that's the waveform panel
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.warning("  Could not detect waveform region; using full image")
        return img

    # Filter to contours that are plausibly waveform panels (>10% image area, wide aspect)
    h_img, w_img = img.shape[:2]
    min_area = 0.10 * h_img * w_img
    candidates = [c for c in contours if cv2.contourArea(c) > min_area]
    if not candidates:
        log.warning("  No large dark region found; using full image")
        return img

    largest = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add small padding
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    log.info("  Cropped waveform region: (%d,%d)→(%d,%d)", x1, y1, x2, y2)
    return img[y1:y2, x1:x2]


def _correct_perspective(img: np.ndarray) -> np.ndarray:
    """Detect paper corners and apply homographic warp to straighten."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return img
    pts_src = _order_points(approx.reshape(4, 2).astype(np.float32))
    h, w = img.shape[:2]
    pts_dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img, M, (w, h))


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


# ---------------------------------------------------------------------------
# Step 2 — Trace isolation
# ---------------------------------------------------------------------------

def isolate_trace(img_bgr: np.ndarray, source: SourceType) -> np.ndarray:
    """
    Returns binary mask (uint8, 255 = trace).
    Screen sources: targeted HSV mask for bright green trace.
    Paper sources: LAB k-means (color) or adaptive Otsu (black ink).
    """
    log.info("Step 2: Isolating ECG trace (%s)", source.name)

    if source in (SourceType.SCREENSHOT, SourceType.VIDEO):
        mask = _hsv_trace_mask(img_bgr)
    else:
        mask = _paper_trace_mask(img_bgr)

    log.info("  Trace coverage: %.1f%%", 100 * mask.sum() / mask.size)
    return mask


def _hsv_trace_mask(img: np.ndarray) -> np.ndarray:
    """
    Targeted HSV threshold for bright-green-on-black screen ECGs.
    Green trace: Hue ~60° (OpenCV: 30), high Saturation, high Value.
    Also handles white/cyan traces by broadening the Value range.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Bright green (most digital ECG monitors)
    green_lo = np.array([25, 80, 80],  dtype=np.uint8)
    green_hi = np.array([95, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, green_lo, green_hi)

    # White/near-white trace fallback (some monochrome displays)
    white_lo = np.array([0, 0, 200], dtype=np.uint8)
    white_hi = np.array([180, 40, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, white_lo, white_hi)

    mask = cv2.bitwise_or(mask_green, mask_white)

    # Unsharp mask to recover blurred edges
    l = img[:, :, 0].astype(np.float32)   # use B channel proxy; not critical here
    return mask


def _paper_trace_mask(img: np.ndarray) -> np.ndarray:
    """LAB k-means for color ink; adaptive Otsu for black ink."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Sharpen edges before thresholding
    l = lab[:, :, 0].astype(np.float32)
    l_blur = cv2.GaussianBlur(l, (0, 0), 1.0)
    lab[:, :, 0] = np.clip(cv2.addWeighted(l, 1.8, l_blur, -0.8, 0), 0, 255).astype(np.uint8)

    is_chromatic = (float(np.std(lab[:, :, 1])) > 8.0 or float(np.std(lab[:, :, 2])) > 8.0)

    if is_chromatic:
        return _kmeans_segment(lab, dark_trace=True)
    else:
        return _adaptive_otsu_segment(lab[:, :, 0])


def _kmeans_segment(lab: np.ndarray, dark_trace: bool = True) -> np.ndarray:
    """k-means (k=3) cluster; select trace by darkest+chromatic (paper) or brightest+chromatic (screen)."""
    h, w = lab.shape[:2]
    scale = 0.25
    small = cv2.resize(lab, (int(w * scale), int(h * scale)))
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    chroma = np.sqrt(centers[:, 1]**2 + centers[:, 2]**2)
    # Paper: trace is dark + chromatic → score = -L + chroma
    # Screen: trace is bright + chromatic → score = L + chroma
    sign = -1 if dark_trace else 1
    scores = sign * centers[:, 0] + chroma
    trace_cluster = int(np.argmax(scores))

    pixels_full = lab.reshape(-1, 3).astype(np.float32)
    _, labels_full, _ = cv2.kmeans(pixels_full, 3, None, criteria, 3,
                                   cv2.KMEANS_USE_INITIAL_LABELS,
                                   bestLabels=np.tile(labels,
                                       (int(1/scale**2)+1, 1)).flatten()[:h*w].reshape(-1,1))
    return (labels_full.flatten() == trace_cluster).reshape(h, w).astype(np.uint8) * 255


def _adaptive_otsu_segment(gray: np.ndarray) -> np.ndarray:
    """4×4 tile-based Otsu to handle uneven paper illumination."""
    th, tw = gray.shape[0] // 4, gray.shape[1] // 4
    mask = np.zeros_like(gray, dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            tile = gray[r*th:(r+1)*th, c*tw:(c+1)*tw]
            _, t = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
    return mask


# ---------------------------------------------------------------------------
# Step 3 — Morphological cleanup
# ---------------------------------------------------------------------------

def clean_trace(
    mask: np.ndarray,
    source: SourceType,
    px_per_div: float = 50.0,
) -> np.ndarray:
    """
    Erode to strip grid/annotation noise, dilate horizontally to rebuild
    trace continuity, then filter out non-trace blobs.
    px_per_div: used to size kernels dynamically.
    """
    log.info("Step 3: Morphological cleanup")

    blurred = cv2.GaussianBlur(mask, (0, 0), 1.2)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Dynamic kernel: erosion strips grid lines (~0.3 div wide)
    ek = max(3, round(0.3 * px_per_div * 0.1))
    ek = ek if ek % 2 == 1 else ek + 1
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (ek, ek))
    eroded = cv2.erode(binary, erode_k, iterations=1)

    # Horizontal dilation rebuilds continuity along signal axis only
    dk = max(7, round(0.7 * px_per_div * 0.1))
    dk = dk if dk % 2 == 1 else dk + 1
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, 1))
    dilated = cv2.dilate(eroded, dilate_k, iterations=1)

    # Drop blobs too small, too tall/narrow, or too isolated to be trace segments
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    clean = np.zeros_like(dilated)
    min_area = 0.003 * mask.size
    for i in range(1, n_labels):
        area  = stats[i, cv2.CC_STAT_AREA]
        w_c   = stats[i, cv2.CC_STAT_WIDTH]
        h_c   = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w_c / (h_c + 1e-6)
        # Traces are wide; annotations/labels are tall or square
        if area >= min_area and aspect > 1.5:
            clean[labels == i] = 255

    log.info("  Blobs retained: %d", int(np.any(clean)))
    return clean


# ---------------------------------------------------------------------------
# Step 4 — Calibration
# ---------------------------------------------------------------------------

def calibrate(
    img_gray: np.ndarray,
    source: SourceType,
    lead_panels: list[tuple[int, int]],     # [(y_start, y_end), ...] per lead
    user_scale: Optional[UserScale] = None,
) -> CalibrationResult:
    """
    PAPER: Hough-based grid detection → median inter-line spacing → mm/px.
    SCREENSHOT/VIDEO: User supplies mV/div and sec/div; panel height gives px/div.
    """
    log.info("Step 4: Calibrating")

    if source == SourceType.PAPER:
        return _calibrate_paper(img_gray)
    else:
        if user_scale is None:
            raise ValueError("UserScale (mv_per_div, seconds_per_div) is required for screen sources")
        return _calibrate_screen(img_gray, user_scale, lead_panels)


def _calibrate_paper(gray: np.ndarray) -> CalibrationResult:
    """Hough transform grid detection with dual 1mm/5mm frequency validation."""
    edges = cv2.Canny(gray, 30, 80)
    tested = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
    h_sp, angles, dists = hough_line(edges, theta=tested)
    _, a_pk, d_pk = hough_line_peaks(h_sp, angles, dists,
                                      min_distance=5, threshold=0.3*h_sp.max())

    horiz = [d for a, d in zip(a_pk, d_pk) if abs(a) < np.radians(10)]
    vert  = [d for a, d in zip(a_pk, d_pk) if abs(a - np.pi/2) < np.radians(10)]

    px_div_x = _spacing_to_px_per_div(vert)
    px_div_y = _spacing_to_px_per_div(horiz)

    # ECG standard: 1 small square = 1mm = 0.1mV = 0.04s
    mv_per_px  = 0.1 / (px_div_y / 5)   # assuming 5 small squares per large div
    sec_per_px = 0.04 / (px_div_x / 5)

    confidence = float(np.clip(
        min(px_div_x, px_div_y) / (max(px_div_x, px_div_y) + 1e-9), 0, 1))

    log.info("  Paper calibration: %.1f px/div X, %.1f px/div Y", px_div_x, px_div_y)
    return CalibrationResult(px_div_x, px_div_y, mv_per_px, sec_per_px,
                             SourceType.PAPER, confidence)


def _spacing_to_px_per_div(positions: list[float]) -> float:
    if len(positions) < 2:
        log.warning("  Too few lines for calibration; defaulting to 50px/div")
        return 50.0
    spacings = np.diff(sorted(positions))
    median_sp = float(np.median(spacings))
    # Validate against ECG standard (5mm large square ≈ 5× small square)
    for n_small_squares in [5, 1]:
        candidate = median_sp / n_small_squares
        if 3 < candidate < 200:  # plausible pixel range
            return candidate * 5  # return px per large div
    return median_sp


def _calibrate_screen(
    gray: np.ndarray,
    scale: UserScale,
    lead_panels: list[tuple[int, int]],
) -> CalibrationResult:
    """
    Derive px/div from detected panel heights and user-supplied mV/div, sec/div.
    Number of vertical divisions assumed from panel height.
    """
    if not lead_panels:
        h = gray.shape[0]
        lead_panels = [(0, h)]

    panel_heights = [y2 - y1 for y1, y2 in lead_panels]
    median_h = float(np.median(panel_heights))

    # Standard ECG display: 8 vertical divisions per panel
    n_divs_y = 8.0
    px_div_y = median_h / n_divs_y

    # Horizontal: infer from aspect ratio (standard 10s strip = 25 div wide)
    w = gray.shape[1]
    n_divs_x = w / px_div_y   # assume square grid divs
    px_div_x = w / n_divs_x

    mv_per_px  = scale.mv_per_div / px_div_y
    sec_per_px = scale.seconds_per_div / px_div_x

    log.info("  Screen calibration: %.1f px/div X, %.1f px/div Y | "
             "%.4f mV/px, %.5f s/px", px_div_x, px_div_y, mv_per_px, sec_per_px)

    return CalibrationResult(px_div_x, px_div_y, mv_per_px, sec_per_px,
                             SourceType.SCREENSHOT, confidence=1.0)


# ---------------------------------------------------------------------------
# Step 4b — Auto-detect lead panel boundaries
# ---------------------------------------------------------------------------

def detect_lead_panels(
    img_bgr: np.ndarray,
    source: SourceType,
    n_leads_hint: Optional[int] = None,
) -> list[tuple[int, int, str]]:
    """
    Find horizontal band boundaries separating individual lead rows.
    Returns [(y_start, y_end, lead_name), ...].
    """
    log.info("Step 4b: Detecting lead panel boundaries")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if source == SourceType.PAPER:
        # For paper, use horizontal projection of the cleaned trace mask
        thresh_val = 200
    else:
        # For screen, dark rows between panels are near-black
        thresh_val = 30

    row_means = np.mean(gray < thresh_val, axis=1)   # fraction of dark pixels per row

    # Rows where almost all pixels are background = panel separator
    is_separator = row_means > 0.90
    # Morphologically close small gaps in separators
    is_separator = ndimage.binary_closing(is_separator, structure=np.ones(10))

    # Find transitions to get panel extents
    transitions = np.diff(is_separator.astype(int))
    starts = list(np.where(transitions == -1)[0] + 1)   # separator → content
    ends   = list(np.where(transitions == 1)[0])         # content → separator

    # Handle edge cases
    if not starts:
        starts = [0]
    if not ends or ends[-1] < starts[-1]:
        ends.append(h)

    panels = list(zip(starts, ends))

    # Filter panels that are too small (likely annotation rows, not lead rows)
    min_panel_h = h * 0.08
    panels = [(y1, y2) for y1, y2 in panels if (y2 - y1) >= min_panel_h]

    # Assign standard lead names
    standard_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    named = []
    for i, (y1, y2) in enumerate(panels):
        name = standard_names[i] if i < len(standard_names) else f"Lead_{i+1}"
        named.append((y1, y2, name))

    log.info("  Detected %d lead panels", len(named))
    for y1, y2, name in named:
        log.info("    %s: rows %d–%d", name, y1, y2)

    return named


# ---------------------------------------------------------------------------
# Step 5 — Pixel-to-voltage conversion
# ---------------------------------------------------------------------------

def extract_lead_signals(
    mask: np.ndarray,
    calibration: CalibrationResult,
    lead_rows: list[tuple[int, int, str]],
    sample_rate_hz: float = 500.0,
) -> list[LeadSignal]:
    """
    Convert mask pixels to calibrated voltage per lead.
    Screen images: bright-on-dark → topmost bright pixel = highest voltage (no sign flip needed).
    Paper images: dark-on-light → invert.
    """
    log.info("Step 5: Converting pixels to voltage")
    leads = []

    for y_start, y_end, name in lead_rows:
        row_mask = mask[y_start:y_end, :]
        w = row_mask.shape[1]

        voltage_px = np.full(w, np.nan)
        for col in range(w):
            col_data = row_mask[:, col]
            trace_rows = np.where(col_data > 0)[0]
            if len(trace_rows):
                # Use centroid rather than topmost pixel for sub-pixel accuracy
                voltage_px[col] = float(np.mean(trace_rows))

        nan_mask = np.isnan(voltage_px)
        continuity_pct = 100.0 * (1 - nan_mask.sum() / w)

        if nan_mask.all():
            log.warning("  %s: entirely missing — skipping", name)
            continue

        # Interpolate gaps
        idx = np.arange(w)
        voltage_px = np.interp(idx, idx[~nan_mask], voltage_px[~nan_mask])

        # Rolling median baseline (200ms window)
        window = max(3, int(0.2 * sample_rate_hz))
        baseline_px = ndimage.median_filter(voltage_px, size=window, mode="nearest")

        # Convert to mV
        # Negate: image y increases downward, voltage increases upward
        voltage_mv = -(voltage_px - baseline_px) * calibration.mv_per_px / 0.1 * 0.1
        # Simplified: voltage_mv = -(y - baseline) * mv_per_px
        voltage_mv = -(voltage_px - baseline_px) * calibration.mv_per_px

        baseline_drift_mv = (-(baseline_px - np.median(baseline_px))
                             * calibration.mv_per_px)

        # For video: additional high-pass filter to remove monitor brightness flicker
        if calibration.source_type == SourceType.VIDEO:
            b, a = signal.butter(2, 0.5 / (sample_rate_hz / 2), btype='high')
            voltage_mv = signal.filtfilt(b, a, voltage_mv)

        time_s = np.arange(w) * calibration.sec_per_px
        snr = _compute_snr(voltage_mv)

        leads.append(LeadSignal(name, voltage_mv, time_s, snr,
                                continuity_pct, baseline_drift_mv))
        log.info("  %s — SNR: %.1f dB, continuity: %.1f%%", name, snr, continuity_pct)

    return leads


def _compute_snr(v: np.ndarray) -> float:
    """SNR on QRS-gated windows: 20·log10(P_signal / P_noise)."""
    if v.max() == v.min():
        return 0.0
    sig_mask = np.abs(v) > 0.3 * np.abs(v).max()
    p_sig  = float(np.mean(v[sig_mask]**2))  if sig_mask.any()  else 1e-9
    p_noise = float(np.mean(v[~sig_mask]**2)) if (~sig_mask).any() else 1e-12
    return float(20 * np.log10(np.sqrt(p_sig) / (np.sqrt(p_noise) + 1e-12)))


# ---------------------------------------------------------------------------
# Step 6 — Quality control
# ---------------------------------------------------------------------------

# Source-aware SNR floors
SNR_FLOOR = {SourceType.PAPER: 15.0, SourceType.SCREENSHOT: 15.0, SourceType.VIDEO: 10.0}
CONTINUITY_THRESHOLD  = 95.0
QRS_AMP_RANGE_MV      = (0.1, 3.5)
RR_CV_THRESHOLD       = 0.30
HF_ARTIFACT_THRESHOLD = 0.15


def quality_control(
    leads: list[LeadSignal],
    calibration: CalibrationResult,
    sample_rate_hz: float = 500.0,
) -> tuple[bool, list[str]]:
    """Per-lead QC gates. Returns (passed, [structured_rejection_reasons])."""
    log.info("Step 6: Quality control")
    reasons = []

    if not leads:
        return False, ["NO_LEADS: No lead signals extracted"]

    snr_floor = SNR_FLOOR[calibration.source_type]

    if calibration.confidence < 0.8:
        reasons.append(f"CALIBRATION_ASYMMETRY: confidence={calibration.confidence:.2f} "
                       "(X/Y scale mismatch)")

    for lead in leads:

        if lead.snr_db < snr_floor:
            reasons.append(f"{lead.name} SNR_FAIL: {lead.snr_db:.1f} dB "
                           f"(floor={snr_floor} dB for {calibration.source_type.name})")

        if lead.continuity_pct < CONTINUITY_THRESHOLD:
            reasons.append(f"{lead.name} CONTINUITY_FAIL: {lead.continuity_pct:.1f}%")

        r_peaks, _ = signal.find_peaks(lead.voltage_mv,
                                        height=QRS_AMP_RANGE_MV[0],
                                        distance=int(0.3 * sample_rate_hz))
        if len(r_peaks) == 0:
            reasons.append(f"{lead.name} NO_QRS: No R-peaks above {QRS_AMP_RANGE_MV[0]} mV")
            continue

        peak_amps = lead.voltage_mv[r_peaks]
        if np.any(peak_amps > QRS_AMP_RANGE_MV[1]):
            reasons.append(f"{lead.name} AMP_OVERFLOW: {peak_amps.max():.2f} mV "
                           f"> {QRS_AMP_RANGE_MV[1]} mV (likely calibration error)")

        if len(r_peaks) >= 3:
            rr = np.diff(r_peaks) / sample_rate_hz
            rr_cv = float(np.std(rr) / (np.mean(rr) + 1e-9))
            if rr_cv > RR_CV_THRESHOLD:
                reasons.append(f"{lead.name} IRREGULAR_RR: CV={rr_cv:.2f} "
                               "(arrhythmia or motion artifact)")

        freqs, psd = signal.welch(lead.voltage_mv, fs=sample_rate_hz,
                                   nperseg=min(256, len(lead.voltage_mv)))
        hf_ratio = float(psd[(freqs>=25)&(freqs<=40)].sum()
                         / (psd[freqs <= sample_rate_hz/2].sum() + 1e-12))
        if hf_ratio > HF_ARTIFACT_THRESHOLD:
            reasons.append(f"{lead.name} MOTION_ARTIFACT: HF ratio={hf_ratio:.2f}")

        if lead.name in ("I", "Lead_I"):
            if float(lead.voltage_mv[r_peaks].mean()) < 0:
                reasons.append(f"{lead.name} POLARITY_INVERTED: possible lead reversal")

    passed = len(reasons) == 0
    status = "PASSED" if passed else f"FAILED ({len(reasons)} issue(s))"
    log.info("  QC %s", status)
    if not passed:
        for r in reasons:
            log.warning("    → %s", r)
    return passed, reasons


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    user_scale: Optional[UserScale] = None,
    is_video_frame: bool = False,
    prev_frames: Optional[list[np.ndarray]] = None,
    sample_rate_hz: float = 500.0,
    plot: bool = True,
) -> ExtractionResult:
    """
    Full ECG extraction pipeline.

    Args:
        image_path:      Path to input image (paper photo, screenshot, or video frame).
        user_scale:      Required for SCREENSHOT/VIDEO: UserScale(mv_per_div, seconds_per_div).
                         e.g. UserScale(mv_per_div=0.5, seconds_per_div=0.2)
        is_video_frame:  Set True when passing individual video frames.
        prev_frames:     Up to 4 previous BGR frames for temporal averaging (video only).
        sample_rate_hz:  Pixel-to-time sample rate (typically 500 Hz equivalent).
        plot:            Display result plot.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    # 1. Normalize
    img_norm, source = normalize_image(img_bgr, is_video_frame, prev_frames)

    # 2. Isolate trace
    trace_mask = isolate_trace(img_norm, source)

    # 3. Initial cleanup with default px_per_div
    clean_mask = clean_trace(trace_mask, source, px_per_div=50.0)

    # 4a. Detect lead panels automatically
    lead_rows = detect_lead_panels(img_norm, source)

    # 4b. Calibrate
    panel_bounds = [(y1, y2) for y1, y2, _ in lead_rows]
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    calibration = calibrate(gray, source, panel_bounds, user_scale)

    # Re-clean with calibrated px_per_div
    clean_mask = clean_trace(trace_mask, source, calibration.px_per_div_y)

    # 5. Extract signals
    leads = extract_lead_signals(clean_mask, calibration, lead_rows, sample_rate_hz)

    # 6. QC
    passed, reasons = quality_control(leads, calibration, sample_rate_hz)

    result = ExtractionResult(leads, calibration, passed, source, reasons)

    if plot:
        _plot_results(img_norm, clean_mask, result)

    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_results(img: np.ndarray, mask: np.ndarray, result: ExtractionResult) -> None:
    n = len(result.leads)
    if n == 0:
        return
    fig, axes = plt.subplots(2 + n, 1, figsize=(14, 4 + 3 * n))
    fig.suptitle(f"ECG Extraction [{result.source_type.name}]  |  "
                 f"QC: {'PASS' if result.passed_qc else 'FAIL'}", fontsize=13)

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Normalized & Cropped Input")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Extracted Trace Mask")
    axes[1].axis("off")

    trace_color = "#00ff00" if result.source_type != SourceType.PAPER else "#1a73e8"
    bg_color    = "#111111" if result.source_type != SourceType.PAPER else "#ffffff"

    for i, lead in enumerate(result.leads):
        ax = axes[2 + i]
        ax.set_facecolor(bg_color)
        ax.plot(lead.time_s, lead.voltage_mv, color=trace_color, linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("mV")
        ax.set_title(f"{lead.name}  |  SNR={lead.snr_db:.1f} dB  |  "
                     f"Continuity={lead.continuity_pct:.1f}%")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "ecg_sample.jpg"

    # For paper printouts — no user_scale needed
    # result = run_pipeline(image_path)

    # For digital screenshots (like the demo images):
    result = run_pipeline(
        image_path=image_path,
        user_scale=UserScale(mv_per_div=0.5, seconds_per_div=0.2),
        sample_rate_hz=500.0,
        plot=True,
    )

    print("\n=== Extraction Summary ===")
    print(f"Source    : {result.source_type.name}")
    print(f"QC Status : {'PASSED' if result.passed_qc else 'FAILED'}")
    print(f"Calibration: {result.calibration.mv_per_px*1000:.3f} µV/px, "
          f"{result.calibration.sec_per_px*1000:.4f} ms/px")
    for lead in result.leads:
        print(f"  {lead.name:6s}: SNR={lead.snr_db:.1f} dB, "
              f"continuity={lead.continuity_pct:.1f}%, "
              f"peak={lead.voltage_mv.max():.2f} mV")
    if result.rejection_reasons:
        print("\nRejection Reasons:")
        for r in result.rejection_reasons:
            print(f"  ✗ {r}")