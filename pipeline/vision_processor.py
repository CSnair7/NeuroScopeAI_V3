"""
Module A — Vision Processor
Camera frame → trace mask + calibration.

Refactored from extraction_pipeline.py (Steps 1–3) with theme-aware
segmentation for different ECG monitor backgrounds.

Changes vs original:
  - Added dark_monitor_green ThemePreset (neon green on black screens)
  - Added _boost_low_light() gamma correction before normalization
  - Added _mask_glare() specular highlight exclusion
  - Added _correct_skew() rotation correction for handheld use
  - Fixed CLAHE: only applied on bright (paper) backgrounds
  - Fixed column valley threshold: percentile-based instead of fixed ratio
  - Fixed perspective warp: skipped when image is likely a monitor screen
  - Lead naming made configurable per monitor profile
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

from config import ThemePreset, get_theme_preset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# scipy/skimage-free helpers
# ---------------------------------------------------------------------------

def _gaussian_filter1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing via cv2.GaussianBlur."""
    if sigma <= 0 or len(arr) == 0:
        return arr.copy()
    radius = max(1, int(round(3 * sigma)))
    ksize = 2 * radius + 1
    row = arr.astype(np.float32).reshape(1, -1)
    out = cv2.GaussianBlur(row, (ksize, 1), sigmaX=sigma, sigmaY=0,
                           borderType=cv2.BORDER_REPLICATE)
    return out.reshape(-1)


def _binary_closing_1d(arr: np.ndarray, size: int) -> np.ndarray:
    """
    Binary closing on a 1-D boolean array (replaces scipy.ndimage.binary_closing).
    Equivalent to morphological dilation followed by erosion.
    """
    if size < 1:
        return arr.copy()
    uint = arr.astype(np.uint8).reshape(1, -1)
    k = np.ones((1, size), dtype=np.uint8)
    dilated = cv2.dilate(uint, k, iterations=1)
    closed  = cv2.erode(dilated, k, iterations=1)
    return closed.reshape(-1).astype(bool)



# ---------------------------------------------------------------------------
# Built-in theme presets
# ---------------------------------------------------------------------------

BUILTIN_THEMES: Dict[str, ThemePreset] = {
    # Neon green trace on black/dark monitor background (most common ICU/OR monitor)
    "dark_monitor_green": ThemePreset(
        name="dark_monitor_green",
        display_name="Dark Monitor — Green Trace",
        # Widened for real camera captures where glare and blur desaturate
        # and shift the green trace. Through a camera pointed at a monitor:
        #   - H can shift from 40-85 to as low as 25 (yellowish-green)
        #   - S drops to 20-30 due to glare wash and ambient reflections
        #   - V drops to 35-50 in areas with partial glare overlay
        hsv_lower=(25, 20, 35),
        hsv_upper=(95, 255, 255),
        trace_is_bright=True,
        bg_hsv_lower=(0, 0, 0),
        bg_hsv_upper=(180, 60, 50),
    ),
    # Black trace on bright pink/red grid paper (standard printed ECG)
    "paper_dark_trace": ThemePreset(
        name="paper_dark_trace",
        display_name="Paper ECG — Dark Trace",
        hsv_lower=(0, 0, 0),
        hsv_upper=(180, 50, 80),
        trace_is_bright=False,
        bg_hsv_lower=(0, 10, 180),
        bg_hsv_upper=(20, 80, 255),
    ),
    # Yellow/orange trace on dark monitor (some older Philips / GE monitors)
    "dark_monitor_yellow": ThemePreset(
        name="dark_monitor_yellow",
        display_name="Dark Monitor — Yellow/Orange Trace",
        hsv_lower=(15, 80, 80),
        hsv_upper=(40, 255, 255),
        trace_is_bright=True,
        bg_hsv_lower=(0, 0, 0),
        bg_hsv_upper=(180, 60, 50),
    ),
    # White trace on dark background (some defibrillator / transport monitors)
    "dark_monitor_white": ThemePreset(
        name="dark_monitor_white",
        display_name="Dark Monitor — White Trace",
        hsv_lower=(0, 0, 180),
        hsv_upper=(180, 40, 255),
        trace_is_bright=True,
        bg_hsv_lower=(0, 0, 0),
        bg_hsv_upper=(180, 60, 50),
    ),
    # Blue trace on dark background (some Mindray / Nihon Kohden monitors)
    "dark_monitor_blue": ThemePreset(
        name="dark_monitor_blue",
        display_name="Dark Monitor — Blue Trace",
        hsv_lower=(95, 80, 80),
        hsv_upper=(130, 255, 255),
        trace_is_bright=True,
        bg_hsv_lower=(0, 0, 0),
        bg_hsv_upper=(180, 60, 50),
    ),
    # Blue trace on white/light background (matplotlib default, some paper ECGs)
    "blue_on_white": ThemePreset(
        name="blue_on_white",
        display_name="Blue on White",
        # Matplotlib default blue: roughly HSV 210-240° → OpenCV hue 105-120
        hsv_lower=(100, 60, 50),
        hsv_upper=(130, 255, 230),
        trace_is_bright=False,
        bg_hsv_lower=(0, 0, 180),
        bg_hsv_upper=(180, 40, 255),
    ),
}

# Standard 12-lead grid naming profiles keyed by (n_cols, max_rows_per_col)
LEAD_NAME_PROFILES: Dict[str, List[List[str]]] = {
    "2col_6row": [
        ["Lead_I", "Lead_II", "Lead_III", "aVR", "aVL", "aVF"],
        ["V1", "V2", "V3", "V4", "V5", "V6"],
    ],
    "4col_3row": [
        ["Lead_I", "Lead_II", "Lead_III"],
        ["aVR", "aVL", "aVF"],
        ["V1", "V2", "V3"],
        ["V4", "V5", "V6"],
    ],
    "3col_4row": [
        ["Lead_I", "Lead_II", "Lead_III", "aVR"],
        ["aVL", "aVF", "V1", "V2"],
        ["V3", "V4", "V5", "V6"],
    ],
}

FALLBACK_LEAD_NAMES = [
    "Lead_I", "Lead_II", "Lead_III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


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
    # 5-tuple: (y_start, y_end, x_start, x_end, name) — all in full-image coords.
    # x_start/x_end define the column bounds of each cell so that extract_leads
    # can crop mask[y:y, x:x] to exactly one lead, preventing side-by-side leads
    # from being merged into a single wide strip.
    lead_rows: List[Tuple[int, int, int, int, str]] = field(default_factory=list)
    glare_mask: Optional[np.ndarray] = None   # uint8 mask: 255=valid, 0=glare
    ecg_roi: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) of ECG display in full-image coords

    def get_roi_crop(self, img: np.ndarray) -> np.ndarray:
        """Crop any same-size image (e.g. trace_mask, normalized_image) to the ECG ROI."""
        x, y, w, h = self.ecg_roi
        if w == 0 or h == 0:
            return img
        return img[y:y+h, x:x+w]

    @property
    def trace_coverage(self) -> float:
        """Fraction of pixels that are active trace (0.0–1.0).
        If this is >0 but downstream signal extraction returns nothing,
        the bug is in the signal extractor, not the vision processor.
        Use this to distinguish the two failure modes."""
        if self.trace_mask is None:
            return 0.0
        return float(self.trace_mask.sum()) / float(self.trace_mask.size * 255)

    @property
    def is_blank(self) -> bool:
        """True if the trace mask contains no signal pixels."""
        return self.trace_coverage == 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _auto_select_theme(img_bgr: np.ndarray, current_theme: ThemePreset) -> ThemePreset:
    """
    Inspect the image and return the best-matching ThemePreset.

    Detection logic:
      1. Measure median background luminance in the image.
         Dark background (< 0.4) → monitor.  Light background (≥ 0.4) → paper.
      2. On light background: check whether dominant chromatic color is blue
         (matplotlib/some paper ECGs) or achromatic dark (standard print).
      3. On dark background: check whether trace is green (default monitor),
         yellow, or white.  If current_theme already matches, keep it.

    This runs in <5 ms on a 720p image (no heavy ops, just HSV histogram).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
    bg_luminance = float(np.median(v_channel))

    is_light_bg = bg_luminance > 0.4

    if is_light_bg:
        # Count pixels in blue hue range (OpenCV hue 100-130 = 200-260° ≈ blue)
        blue_mask = cv2.inRange(hsv,
                                np.array([100, 50, 40],  dtype=np.uint8),
                                np.array([130, 255, 230], dtype=np.uint8))
        blue_frac = float(blue_mask.sum()) / float(blue_mask.size * 255)

        if blue_frac > 0.002:
            # Significant blue pixels → blue-on-white
            new_theme = get_theme_preset("blue_on_white")
            if new_theme.name != current_theme.name:
                log.info("  Auto-theme: light background + blue trace → blue_on_white "
                         "(was %s)", current_theme.name)
            return new_theme
        else:
            # Light background, no strong blue → black trace on white paper
            new_theme = get_theme_preset("black_on_white")
            if new_theme.name != current_theme.name:
                log.info("  Auto-theme: light background, no blue → black_on_white "
                         "(was %s)", current_theme.name)
            return new_theme
    else:
        # Dark background monitor — check dominant trace hue
        # Count green pixels (neon green: hue 35-85)
        green_mask = cv2.inRange(hsv,
                                 np.array([35, 80, 80],  dtype=np.uint8),
                                 np.array([85, 255, 255], dtype=np.uint8))
        green_frac = float(green_mask.sum()) / float(green_mask.size * 255)

        if green_frac > 0.002:
            new_theme = get_theme_preset("green_on_black")
            if new_theme.name != current_theme.name:
                log.info("  Auto-theme: dark background + green trace → green_on_black "
                         "(was %s)", current_theme.name)
            return new_theme

        # Check yellow
        yellow_mask = cv2.inRange(hsv,
                                  np.array([15, 80, 80],  dtype=np.uint8),
                                  np.array([40, 255, 255], dtype=np.uint8))
        yellow_frac = float(yellow_mask.sum()) / float(yellow_mask.size * 255)
        if yellow_frac > 0.005:
            new_theme = get_theme_preset("yellow_on_dark")
            if new_theme.name != current_theme.name:
                log.info("  Auto-theme: dark background + yellow trace → yellow_on_dark "
                         "(was %s)", current_theme.name)
            return new_theme

        # Default: keep current theme (handles white_on_black etc.)
        return current_theme


def process_frame(
    frame_bgr: np.ndarray,
    theme: Optional[ThemePreset] = None,
    theme_name: str = "dark_monitor_green",
    skip_perspective: bool = True,
    expected_leads: int = 0,
    lead_profile: Optional[str] = None,
) -> VisionResult:
    """
    End-to-end vision processing for a single camera frame.

    Args:
        frame_bgr:        Raw BGR image (from camera or file).
        theme:            Active ThemePreset. If None, uses theme_name lookup.
        theme_name:       Name of a built-in theme preset (used when theme=None).
                          Default is "dark_monitor_green" for ICU/OR monitor screens.
        skip_perspective: If True, skip perspective correction. Recommended True
                          for monitor screens to avoid bezel-detection false warps.
        expected_leads:   Number of leads the user expects (0 = auto-detect).
        lead_profile:     Key into LEAD_NAME_PROFILES, e.g. "2col_6row".
                          None = auto-select from detected grid shape.

    Returns:
        VisionResult with trace mask, spatial calibration, and lead rows.
    """
    t0 = time.perf_counter()

    # Resolve theme — with auto-detection override.
    # If the caller passes a fixed theme, honour it.  Otherwise inspect the
    # image to pick the best-matching preset before running the pipeline.
    if theme is None:
        theme = BUILTIN_THEMES.get(theme_name) or get_theme_preset(theme_name)

    # Auto-detect theme when the resolved theme doesn't match the image.
    # Strategy: measure background luminance and dominant trace hue to
    # choose between dark-monitor (green/yellow/white trace on black) and
    # paper (black or blue trace on white).
    theme = _auto_select_theme(frame_bgr, theme)

    # --- Pre-processing ---

    # Step 0a — Low-light gamma boost (before any other processing)
    img_boosted = _boost_low_light(frame_bgr)

    # Step 0b — Compute glare mask (before normalization alters pixel values)
    glare_valid = _mask_glare(img_boosted)

    # Step 1 — Normalize (perspective + illumination)
    img_norm = normalize_image(img_boosted, skip_perspective=skip_perspective)

    # Step 2 — Isolate trace using theme-aware segmentation
    trace_mask = isolate_trace(img_norm, theme)

    # Apply glare mask to remove false positives from reflections
    trace_mask = cv2.bitwise_and(trace_mask, glare_valid)

    # Step 3 — Morphological cleanup (first pass with default mm/px)
    clean_mask = clean_trace(trace_mask, mm_per_px=0.1)

    # Step 4 — Grid calibration (Hough-based, then spectral fallback)
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    mm_x, mm_y, confidence = calibrate_grid(gray)

    # Spectral fallback when Hough confidence is low
    if confidence < 0.4:
        try:
            from pipeline.signal_processor import spectral_grid_estimation
            grid_h_px, grid_v_px = spectral_grid_estimation(gray)
            if grid_h_px > 2 and grid_v_px > 2:
                for grid_mm in [1.0, 5.0]:
                    ratio_x = grid_mm / grid_v_px
                    ratio_y = grid_mm / grid_h_px
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
        clean_mask = cv2.bitwise_and(clean_mask, glare_valid)

    # Sanity-clamp mm_per_px values.
    # On dark monitors with no physical grid lines, Hough calibration detects
    # QRS edges or bezel lines and can return mm_per_px_y >> 0.3 mm/px.
    # A value of 1.0 mm/px would multiply every pixel displacement by 10x,
    # producing a ~10x amplitude overestimate.
    # Physical constraint: at typical monitor resolutions (720-1080p) and
    # standard ECG display gains (10 mm/mV), a 1 mV peak uses 10–30 px of
    # strip height, giving mm_per_px_y ≈ 0.03–0.15.  Cap at 0.3 mm/px to
    # prevent catastrophic over-scaling while allowing for low-res captures.
    MM_PER_PX_MAX = 0.3
    MM_PER_PX_MIN = 0.01
    if mm_x > MM_PER_PX_MAX or mm_x < MM_PER_PX_MIN:
        log.info("  mm_per_px_x %.4f out of range — clamping to [%.2f, %.2f]",
                 mm_x, MM_PER_PX_MIN, MM_PER_PX_MAX)
        mm_x = float(np.clip(mm_x, MM_PER_PX_MIN, MM_PER_PX_MAX))
    if mm_y > MM_PER_PX_MAX or mm_y < MM_PER_PX_MIN:
        log.info("  mm_per_px_y %.4f out of range — clamping to [%.2f, %.2f]",
                 mm_y, MM_PER_PX_MIN, MM_PER_PX_MAX)
        mm_y = float(np.clip(mm_y, MM_PER_PX_MIN, MM_PER_PX_MAX))

    # Step 5 — Detect lead rows
    lead_rows = detect_lead_rows(
        img_norm,
        expected_leads=expected_leads,
        lead_profile=lead_profile,
    )

    latency = (time.perf_counter() - t0) * 1000.0

    # Determine the ECG display ROI (same logic as detect_lead_rows uses)
    ecg_roi = find_ecg_roi(img_norm)

    return VisionResult(
        trace_mask=clean_mask,
        normalized_image=img_norm,
        mm_per_px_x=mm_x,
        mm_per_px_y=mm_y,
        calibration_confidence=confidence,
        latency_ms=latency,
        lead_rows=lead_rows,
        glare_mask=glare_valid,
        ecg_roi=ecg_roi,
    )


# ---------------------------------------------------------------------------
# Step 0a — Low-light gamma boost
# ---------------------------------------------------------------------------

def _boost_low_light(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply gamma correction to underexposed frames so that dim traces
    become visible to downstream segmentation.

    The correction is adaptive: brighter images are left unchanged,
    very dark images get a stronger boost (gamma up to 3.0).
    """
    mean_val = float(img_bgr.mean())

    if mean_val >= 80:
        # Adequately exposed — no correction needed
        return img_bgr

    if mean_val < 1.0:
        mean_val = 1.0

    # Compute gamma so that mean_val maps to ~127
    gamma = np.log(127.0) / np.log(mean_val)
    gamma = float(np.clip(gamma, 1.0, 3.0))

    log.info("  Low-light boost: mean=%.1f → gamma=%.2f", mean_val, gamma)

    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img_bgr, table)


# ---------------------------------------------------------------------------
# Step 0b — Glare masking
# ---------------------------------------------------------------------------

def _mask_glare(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect specular highlights (glare from camera flash / ambient light
    reflecting off monitor glass) and return a validity mask.

    Returns:
        uint8 mask: 255 = valid pixel, 0 = glare region (to be excluded).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Specular glare = high value + low saturation
    # Lowered V threshold from 220→200 and widened S range from 40→60
    # to catch more subtle glare from camera captures (ambient reflections
    # on monitor glass are often not pure white but slightly tinted)
    glare = cv2.inRange(
        hsv,
        np.array([0, 0, 200], dtype=np.uint8),
        np.array([180, 60, 255], dtype=np.uint8),
    )

    # Dilate to cover glare halos and transitional fringe pixels
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    glare_dilated = cv2.dilate(glare, k, iterations=2)

    # Invert: 255 = keep, 0 = glare
    valid = cv2.bitwise_not(glare_dilated)

    glare_frac = glare_dilated.sum() / (glare_dilated.size * 255)
    if glare_frac > 0.0:
        log.info("  Glare mask: %.1f%% of frame excluded", glare_frac * 100)

    return valid


# ---------------------------------------------------------------------------
# Step 1 — Image normalization
# ---------------------------------------------------------------------------

def normalize_image(
    img_bgr: np.ndarray,
    skip_perspective: bool = True,
) -> np.ndarray:
    """
    Correct perspective/skew, even out illumination, and compensate for
    lens distortion. Returns a normalized BGR image.

    Notes:
        - Perspective correction is disabled by default for monitor screens
          because the monitor bezel is detected as the document boundary,
          which warps the waveform area incorrectly.
        - CLAHE illumination normalization is only applied on bright
          (paper-like) backgrounds; on dark monitor screens it amplifies
          noise in the black regions.
        - Skew correction (rotation) is always applied, which handles
          handheld tilt without the risks of full perspective warping.
    """
    h, w = img_bgr.shape[:2]

    # 1a — Lens distortion correction (mild barrel)
    fx = fy = max(h, w) * 0.9
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.05, 0.01, 0, 0, 0], dtype=np.float32)
    img_undist = cv2.undistort(img_bgr, K, dist_coeffs)

    # 1b — Skew (rotation) correction — safe for both paper and monitor
    img_deskewed = _correct_skew(img_undist)

    # 1c — Perspective correction (paper ECGs only; skip for monitors)
    if not skip_perspective:
        img_persp = _correct_perspective(img_deskewed)
    else:
        img_persp = img_deskewed

    # 1d — CLAHE only on bright/paper backgrounds
    #      On dark monitor screens CLAHE amplifies noise in the black field.
    lab = cv2.cvtColor(img_persp, cv2.COLOR_BGR2LAB)
    mean_luminance = float(lab[:, :, 0].mean())

    if mean_luminance > 100:
        # Bright background (paper ECG) — apply CLAHE for illumination leveling
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_persp = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        log.info("  CLAHE applied (mean_lum=%.1f)", mean_luminance)
    else:
        log.info("  CLAHE skipped — dark background (mean_lum=%.1f)",
                 mean_luminance)

    return img_persp


_skew_ema_angle = 0.0   # Module-level persistent EMA for skew angle


def _correct_skew(img: np.ndarray) -> np.ndarray:
    """
    Detect dominant horizontal line angle and rotate to correct handheld tilt.
    Only corrects rotations within ±20° to avoid over-rotating.
    Uses EMA smoothing across frames to prevent jitter.
    """
    global _skew_ema_angle

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=img.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) < 20.0:   # Only near-horizontal lines
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))

    # EMA smooth the detected angle to prevent frame-to-frame jitter
    _SKEW_ALPHA = 0.2   # Low alpha = heavy smoothing
    _skew_ema_angle = _SKEW_ALPHA * median_angle + (1.0 - _SKEW_ALPHA) * _skew_ema_angle

    if abs(_skew_ema_angle) < 1.5:
        return img  # Within tolerance; skip warp

    log.info("  Skew correction: %.2f° (raw=%.2f°)", _skew_ema_angle, median_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), _skew_ema_angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _correct_perspective(img: np.ndarray) -> np.ndarray:
    """
    Detect paper corners and apply homographic warp.
    Only called for paper ECG mode (skip_perspective=False).
    Adds a safety check: only warps if the detected quad covers >40% of
    the image area (avoids warping to monitor bezel or small artifacts).
    """
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

    # Safety: ensure detected quad is a significant portion of the image
    img_area = img.shape[0] * img.shape[1]
    quad_area = float(cv2.contourArea(approx))
    if quad_area / img_area < 0.40:
        log.info("  Perspective correction skipped: quad covers only %.1f%% "
                 "of image", quad_area / img_area * 100)
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

    Pipeline:
      1. Build a background suppression mask using theme.bg_hsv_lower/upper.
         Pixels classified as background are zeroed out before trace detection,
         which removes grid lines and UI chrome that share hue with the trace.
      2. Threshold for trace color using theme.hsv_lower/upper.
      3. Falls back to adaptive segmentation if coverage is out of range.
      4. Unsharp-mask to sharpen trace edges.

    Returns binary mask (uint8, 0 or 255).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- Step 1: suppress known background regions ---
    bg_lower = np.array(theme.bg_hsv_lower, dtype=np.uint8)
    bg_upper = np.array(theme.bg_hsv_upper, dtype=np.uint8)
    bg_mask = cv2.inRange(hsv, bg_lower, bg_upper)

    # Dilate slightly to capture fringe pixels at background/trace boundaries
    bg_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bg_mask_dilated = cv2.dilate(bg_mask, bg_k, iterations=1)

    # Zero out background pixels in HSV before trace thresholding
    hsv_fg = hsv.copy()
    hsv_fg[bg_mask_dilated == 255] = 0

    # --- Step 2: threshold for trace color ---
    lower = np.array(theme.hsv_lower, dtype=np.uint8)
    upper = np.array(theme.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv_fg, lower, upper)

    coverage = float(mask.sum()) / float(mask.size * 255)
    log.info("  HSV threshold coverage: %.2f%% (theme=%s)",
             coverage * 100, theme.name)

    if coverage < 0.005 or coverage > 0.6:
        log.info("  Coverage %.1f%% out of range; falling back to adaptive "
                 "segmentation", coverage * 100)
        mask = _fallback_segment(img_bgr, bg_mask=bg_mask_dilated)

    # --- Step 3: unsharp mask to sharpen trace edges ---
    mask_f = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_f, (0, 0), 1.0)
    sharp = cv2.addWeighted(mask_f, 1.5, blurred, -0.5, 0)
    mask = np.clip(sharp, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def _fallback_segment(
    img_bgr: np.ndarray,
    bg_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Adaptive fallback segmentation using LAB color space.
    Handles unknown or misidentified themes.

    Args:
        img_bgr: Normalized BGR image.
        bg_mask: Optional uint8 mask (255 = background) derived from
                 theme.bg_hsv bounds. When provided, background pixels are
                 zeroed out before clustering/thresholding so that known-BG
                 regions never pollute the trace cluster.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Zero out confirmed background pixels so they don't bias clustering
    if bg_mask is not None:
        lab_masked = lab.copy()
        lab_masked[bg_mask == 255] = 0
    else:
        lab_masked = lab

    a_std = float(np.std(lab_masked[:, :, 1]))
    b_std = float(np.std(lab_masked[:, :, 2]))

    if a_std > 8.0 or b_std > 8.0:
        result = _kmeans_segment(lab_masked)
    else:
        result = _adaptive_otsu_segment(lab_masked[:, :, 0])

    # Ensure background pixels are excluded from the final mask
    if bg_mask is not None:
        result = cv2.bitwise_and(result, cv2.bitwise_not(bg_mask))

    return result


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

    # Connected component filtering — keep signal-like blobs, remove isolated dots
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )
    # min_area: 0.5% was too large for multi-lead images where each lead's trace
    # is a small fraction of the full image. Use 0.05% instead — large enough to
    # reject single-pixel noise but small enough to keep individual QRS segments.
    min_area = 0.0005 * mask.size
    clean = np.zeros_like(dilated)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_comp = stats[i, cv2.CC_STAT_WIDTH]
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        # Removed aspect ratio filter: the original w/h > 2.0 requirement only
        # kept horizontal blobs, which excluded tall narrow QRS spikes (aspect ~0.1-0.4)
        # that are the most diagnostically important features of the ECG.
        # Only filter on minimum area to remove isolated noise pixels.
        if area >= min_area:
            clean[labels == i] = 255

    return clean


# ---------------------------------------------------------------------------
# Step 4.5 — Automatic lead row detection
# ---------------------------------------------------------------------------

def find_ecg_roi(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Locate the ECG display rectangle within a larger image.

    This handles the common case where the input is a full phone screenshot
    containing app chrome, debug panels, and result cards surrounding the
    actual ECG monitor/display area.

    Strategy: find the largest contiguous bounding box of green-hued pixels
    (the ECG trace color). A large morphological close bridges gaps between
    individual waveform segments to produce one solid ROI blob.

    Returns:
        (x, y, w, h) of the ECG display ROI in full-image coordinates.
        Returns (0, 0, img_w, img_h) if no clear ROI is found (safe fallback).
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Cast a wide net: any greenish hue with some saturation and brightness
    green = cv2.inRange(
        hsv,
        np.array([25, 30, 30], dtype=np.uint8),
        np.array([100, 255, 255], dtype=np.uint8),
    )

    # Large close kernel merges all lead strips into one display-area blob
    close_w = min(80, w // 8)
    close_h = min(80, h // 8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_w, close_h)
    )
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if n_labels < 2:
        log.info("  ECG ROI: no green components found, using full image")
        return (0, 0, w, h)

    # Pick the largest component — that's the ECG display area
    areas = stats[1:, cv2.CC_STAT_AREA]
    ml = int(np.argmax(areas)) + 1
    roi_x = int(stats[ml, cv2.CC_STAT_LEFT])
    roi_y = int(stats[ml, cv2.CC_STAT_TOP])
    roi_w = int(stats[ml, cv2.CC_STAT_WIDTH])
    roi_h = int(stats[ml, cv2.CC_STAT_HEIGHT])

    # Sanity check: ROI must be a meaningful fraction of the image
    if roi_w < w * 0.15 or roi_h < h * 0.10:
        log.info("  ECG ROI too small (%.0f%%×%.0f%%), using full image",
                 roi_w/w*100, roi_h/h*100)
        return (0, 0, w, h)

    log.info("  ECG ROI: x=%d y=%d w=%d h=%d (%.0f%%×%.0f%% of image)",
             roi_x, roi_y, roi_w, roi_h, roi_w/w*100, roi_h/h*100)
    return (roi_x, roi_y, roi_w, roi_h)


def detect_lead_rows(
    img_bgr: np.ndarray,
    min_leads: int = 1,
    max_leads: int = 12,
    expected_leads: int = 0,
    lead_profile: Optional[str] = None,
) -> List[Tuple[int, int, int, int, str]]:
    """
    Automatically detect lead strips in single or multi-lead ECG images.

    Handles both:
      - Simple row-band layouts (1-3 leads stacked vertically)
      - 2D grid layouts (12-lead monitor: 2 cols × 6 rows or 4 cols × 3 rows)

    Key behaviour: first crops to the ECG display ROI (largest green-active
    bounding box) before running detection. This prevents app chrome, debug
    panels, titles, and result cards from polluting the activity profile and
    producing wrong row boundaries. Returned coordinates are always in
    full-image space.

    Args:
        img_bgr:        Normalized BGR image (may be full phone screenshot).
        min_leads:      Minimum expected lead count.
        max_leads:      Maximum expected lead count.
        expected_leads: User-specified lead count (0 = auto-detect).
        lead_profile:   Override lead naming profile key (None = auto).

    Returns:
        List of (y_start, y_end, lead_name) in full-image pixel coordinates.
    """
    h, w = img_bgr.shape[:2]

    # Step 0: locate the ECG display within the image
    roi_x, roi_y, roi_w, roi_h = find_ecg_roi(img_bgr)
    roi_img = img_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    def _offset_grid(rows):
        """Lift 5-tuple grid results from ROI-relative to full-image coords.
        Grid cells already carry x_start/x_end (ROI-relative); add roi offsets."""
        return [
            (y_s + roi_y, y_e + roi_y, x_s + roi_x, x_e + roi_x, name)
            for y_s, y_e, x_s, x_e, name in rows
        ]

    def _offset_bands(rows):
        """Lift 3-tuple row-band results to full-image 5-tuples.
        Row-band detection produces full-ROI-width strips, so x spans the ROI."""
        return [
            (y_s + roi_y, y_e + roi_y, roi_x, roi_x + roi_w, name)
            for y_s, y_e, name in rows
        ]

    # Try grid layout detection on the ROI
    grid_result = _detect_grid_layout(roi_img, expected_leads,
                                       lead_profile=lead_profile)
    if grid_result is not None and len(grid_result) >= 4:
        log.info("  Detected grid layout with %d leads", len(grid_result))
        return _offset_grid(grid_result)

    # Fall back to row-band detection on the ROI
    row_result = _detect_row_bands(roi_img, min_leads, max_leads,
                                    expected_leads, lead_profile=lead_profile)
    return _offset_bands(row_result)


def _detect_grid_layout(
    img_bgr: np.ndarray,
    expected_leads: int = 0,
    lead_profile: Optional[str] = None,
) -> Optional[List[Tuple[int, int, str]]]:
    """
    Detect a 2D grid of ECG lead cells (e.g., 12-lead monitor display).

    Strategy:
      1. Build a color-agnostic trace pixel mask (works for green/blue/black traces)
      2. Morphological closing to connect nearby waveform clusters
      3. Find the largest connected component (the waveform grid area)
      4. Within that area, split into columns and rows using density valleys
      5. Validate that all detected columns are roughly equal in width
         (rejects axis-label strips and other narrow UI artefacts)
      6. Require all columns to contain multiple row-bands (true grid cells)
      7. Require minimum cell count to match a plausible ECG grid layout
      8. Assign standard lead names based on grid position

    Key fixes vs original:
      - Percentile-based valley threshold (was fixed fraction of median)
      - Column width uniformity check: all cols must be within 3× of each other.
        Axis-label strips on the left (e.g., "LEAD I" text) are typically
        10-15% of image width while waveform columns are 40-50%, so they
        are cleanly rejected by this ratio test.
      - Minimum column width: each column must be at least 15% of image width.
        This alone filters out tick/label strips on stacked-row layouts.
      - All columns must have the same number of row splits (consistent grid).
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Build a color-agnostic trace pixel mask.
    # We need to find waveform pixels regardless of their color — the image
    # could be green-on-black (monitor), blue-on-white (matplotlib/paper),
    # or black-on-white (printed ECG).
    #
    # Strategy: detect the dominant background and mask everything that ISN'T
    # background.  This works for any trace color.
    #
    # 1. Dark background: anything brighter than the background
    v_channel = hsv[:, :, 2]
    median_v  = float(np.median(v_channel))
    is_light_bg = median_v > 100  # >0.39 on 0-255 scale

    if is_light_bg:
        # Light background: trace pixels are darker (printed/matplotlib).
        # Include chromatic pixels (saturated, any hue) for blue/colored traces
        # AND dark achromatic pixels (black/dark-gray printed traces).
        # Background is white (V>180, S<40) — exclude it.
        bg_mask_light = cv2.inRange(hsv,
                                    np.array([0,   0, 180], dtype=np.uint8),
                                    np.array([180, 40, 255], dtype=np.uint8))
        # Blue/colored traces: S > 50 (chromatic, any hue)
        chroma_mask = cv2.inRange(hsv,
                                  np.array([0,  50,  30], dtype=np.uint8),
                                  np.array([180, 255, 230], dtype=np.uint8))
        # Dark printed traces: V < 100
        dark_mask = cv2.inRange(hsv,
                                np.array([0,  0,   0], dtype=np.uint8),
                                np.array([180, 255, 100], dtype=np.uint8))
        green_mask = cv2.bitwise_or(chroma_mask, dark_mask)
        green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(bg_mask_light))
    else:
        # Dark background monitor: trace pixels are brighter than background.
        # Accept any chromatic pixel (S>40) or bright pixel (V>80).
        chroma_mask = cv2.inRange(hsv,
                                  np.array([0,  40, 60], dtype=np.uint8),
                                  np.array([180, 255, 255], dtype=np.uint8))
        bright_mask = cv2.inRange(hsv,
                                  np.array([0,  0,  100], dtype=np.uint8),
                                  np.array([180, 255, 255], dtype=np.uint8))
        # Exclude the near-black background itself
        bg_dark = cv2.inRange(hsv,
                              np.array([0, 0, 0], dtype=np.uint8),
                              np.array([180, 60, 50], dtype=np.uint8))
        green_mask = cv2.bitwise_or(chroma_mask, bright_mask)
        green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(bg_dark))

    # Morphological closing with large kernel to bridge gaps between lead cells
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find connected components — the largest one is the waveform grid
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    if n_labels < 2:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = int(np.argmax(areas)) + 1
    main_x = stats[main_label, cv2.CC_STAT_LEFT]
    main_y = stats[main_label, cv2.CC_STAT_TOP]
    main_w = stats[main_label, cv2.CC_STAT_WIDTH]
    main_h = stats[main_label, cv2.CC_STAT_HEIGHT]

    if main_w < w * 0.25 or main_h < h * 0.25:
        return None

    x_min, x_max = main_x, main_x + main_w
    y_min, y_max = main_y, main_y + main_h

    log.info("  Grid ROI: x=[%d,%d], y=[%d,%d] (%dx%d)",
             x_min, x_max, y_min, y_max, main_w, main_h)

    green_roi = green_mask[y_min:y_max, x_min:x_max]



    # Column density profile
    col_density = green_roi.astype(np.float32).mean(axis=0) / 255.0
    col_smooth = _gaussian_filter1d(col_density, sigma=max(3, main_w // 60))

    # Percentile-based valley threshold (more robust than fixed ratio of median)
    col_valley_thresh = float(np.percentile(col_smooth, 25))
    col_valleys = col_smooth < col_valley_thresh
    col_splits = _find_splits(col_valleys, min_size=main_w // 20)

    if len(col_splits) < 2:
        log.info("  Grid detection: fewer than 2 column splits found")
        return None

    # --- Validation 1: minimum column width ---
    # Axis-label strips ("LEAD I", tick numbers) are typically <12% of image
    # width. Real ECG waveform columns are always wider. Reject any column
    # narrower than 15% of the full image width.
    min_col_width = int(w * 0.15)
    wide_splits = [(s, e) for s, e in col_splits if (e - s) >= min_col_width]

    if len(wide_splits) < 2:
        log.info("  Grid detection: fewer than 2 columns pass width filter "
                 "(min_col_width=%d px); falling back to row-band", min_col_width)
        return None

    # --- Validation 2: column width uniformity ---
    # In a true multi-column ECG grid all columns are approximately the same
    # width (±50%). A ratio >3× between widest and narrowest indicates that
    # one "column" is actually a margin/label strip.
    col_widths = [e - s for s, e in wide_splits]
    width_ratio = max(col_widths) / (min(col_widths) + 1e-6)
    if width_ratio > 3.0:
        log.info("  Grid detection: column width ratio %.1f× exceeds 3×; "
                 "non-uniform columns suggest label strips — falling back",
                 width_ratio)
        return None

    col_splits = wide_splits
    n_cols = len(col_splits)
    log.info("  Grid: %d valid columns (widths: %s)", n_cols, col_widths)

    # For each column, find row splits
    all_cells = []
    rows_per_col = []
    for ci, (cx_start, cx_end) in enumerate(col_splits):
        col_green = green_mask[y_min:y_max,
                               x_min + cx_start:x_min + cx_end]
        row_dens = col_green.astype(np.float32).mean(axis=1) / 255.0
        row_sm = _gaussian_filter1d(row_dens, sigma=max(3, main_h // 40))

        row_valley_thresh = float(np.percentile(row_sm, 15))
        row_vals = row_sm < row_valley_thresh
        row_splits = _find_splits(row_vals, min_size=main_h // 20)

        if not row_splits:
            row_splits = [(0, main_h)]

        rows_per_col.append(len(row_splits))

        for ri, (ry_start, ry_end) in enumerate(row_splits):
            abs_y_start = y_min + ry_start
            abs_y_end = y_min + ry_end
            all_cells.append((abs_y_start, abs_y_end,
                               x_min + cx_start, x_min + cx_end,
                               ci, ri))

    # --- Validation 3: consistent row count across columns ---
    # In a real grid all columns have the same number of rows.
    # If they differ significantly the "grid" is a false positive.
    if len(set(rows_per_col)) > 1:
        log.info("  Grid detection: inconsistent row counts per column %s; "
                 "not a regular grid — falling back", rows_per_col)
        return None

    # --- Validation 4: minimum cell count for a plausible ECG grid ---
    if len(all_cells) < 6:
        log.info("  Grid detection: only %d cells (<6); falling back",
                 len(all_cells))
        return None

    all_cells.sort(key=lambda c: (c[4], c[5]))

    # Resolve lead naming profile
    if lead_profile and lead_profile in LEAD_NAME_PROFILES:
        name_grid = LEAD_NAME_PROFILES[lead_profile]
    else:
        # Auto-select by detected column count
        profile_key = {2: "2col_6row", 4: "4col_3row", 3: "3col_4row"}.get(n_cols)
        name_grid = LEAD_NAME_PROFILES.get(profile_key) if profile_key else None

    results = []
    for cell in all_cells:
        y_start, y_end, x_start, x_end, ci, ri = cell
        if name_grid and ci < len(name_grid) and ri < len(name_grid[ci]):
            name = name_grid[ci][ri]
        else:
            idx = len(results)
            name = (FALLBACK_LEAD_NAMES[idx]
                    if idx < len(FALLBACK_LEAD_NAMES)
                    else f"Lead_{idx + 1}")

        # Include x_start/x_end so extract_leads crops to exactly this cell.
        # Without x bounds, full-width strips merge side-by-side leads (e.g.
        # Lead I and V1 share the same rows in a 2-col layout), causing the
        # centroid tracker to jump between them and produce a corrupted signal.
        results.append((y_start, y_end, x_start, x_end, name))
        log.info("  Grid cell %s: y=%d-%d x=%d-%d (col %d, row %d)",
                 name, y_start, y_end, x_start, x_end, ci, ri)

    return results


def _find_splits(
    valleys: np.ndarray,
    min_size: int = 10,
) -> List[Tuple[int, int]]:
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
    lead_profile: Optional[str] = None,
) -> List[Tuple[int, int, str]]:
    """
    Detect horizontal lead strips for simple stacked layouts.

    Two-strategy approach:
      Strategy A (threshold): works when inactive gap rows exist between leads.
      Strategy B (valley-split): fallback when the image is uniformly active
        (e.g. a multi-lead monitor where every row has some waveform pixel).

    Uses green-channel variance as the primary activity signal rather than
    gray-mean brightness, which works better on dark-background monitors where
    the trace is a narrow bright line against pure black.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Activity signal ---
    # Green channel variance more reliably separates lead strips from the black
    # inter-strip gaps on dark monitors than gray-mean brightness does.
    green_ch = img_bgr[:, :, 1].astype(np.float32)  # BGR → channel 1 = G
    row_green_mean = green_ch.mean(axis=1)
    row_green_std  = green_ch.std(axis=1)

    g_max = row_green_mean.max() if row_green_mean.max() > 0 else 1.0
    s_max = row_green_std.max()  if row_green_std.max()  > 0 else 1.0
    activity = (row_green_mean / g_max * 0.4 + row_green_std / s_max * 0.6)

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

    # --- Strategy A: threshold-based active-region detection ---
    median_activity = np.median(
        analysis[analysis > np.percentile(analysis, 20)]
    )
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

    # --- Strategy B: valley-split fallback ---
    # When the entire image is uniformly active (all rows contain waveform pixels),
    # Strategy A collapses everything to 1 region. Strategy B instead finds the
    # local minima (darkest separator rows) in the activity profile and splits on
    # them. This correctly handles 12-lead monitors and similar dense layouts.
    if len(regions) <= 1 and analysis.max() > 0.05:
        log.info("  Row-band Strategy A found \u22641 region; "
                 "trying Strategy B (valley-split)")
        fine = _gaussian_filter1d(analysis, sigma=max(2, h // 60))

        # Find local minima
        valley_indices = [
            i for i in range(1, len(fine) - 1)
            if fine[i] < fine[i - 1] and fine[i] < fine[i + 1]
        ]

        if valley_indices:
            val_thresh = float(np.percentile(fine, 40))
            deep_valleys = sorted(
                [i for i in valley_indices if fine[i] < val_thresh]
            )

            if deep_valleys:
                boundaries = [0] + deep_valleys + [len(analysis)]
                candidate_regions = []
                for bi in range(len(boundaries) - 1):
                    y0 = boundaries[bi] + margin
                    y1 = boundaries[bi + 1] + margin
                    if (y1 - y0) >= min_height:
                        candidate_regions.append((y0, y1))

                if len(candidate_regions) > 1:
                    regions = candidate_regions
                    log.info("  Strategy B found %d regions via valley-split",
                             len(regions))

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

    # Resolve naming
    if lead_profile and lead_profile in LEAD_NAME_PROFILES:
        # Flatten profile for row-band use
        flat_names = [
            name
            for col in LEAD_NAME_PROFILES[lead_profile]
            for name in col
        ]
    else:
        flat_names = FALLBACK_LEAD_NAMES

    lead_rows = []
    for i, (y_start, y_end) in enumerate(regions):
        name = flat_names[i] if i < len(flat_names) else f"Lead_{i + 1}"
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
    Detect grid lines and compute mm/px ratios using cv2.HoughLines.

    Returns:
        (mm_per_px_x, mm_per_px_y, confidence)
    """
    try:
        edges = cv2.Canny(img_gray, 30, 80)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
        if lines is None:
            return 0.1, 0.1, 0.0

        horiz, vert = [], []
        for line in lines:
            rho, theta = line[0]
            if abs(theta) < np.radians(10) or abs(theta - np.pi) < np.radians(10):
                horiz.append(abs(rho))
            elif abs(theta - np.pi / 2) < np.radians(10):
                vert.append(abs(rho))

        mm_x = _infer_mm_per_px(sorted(vert))
        mm_y = _infer_mm_per_px(sorted(horiz))

        ratio = min(mm_x, mm_y) / (max(mm_x, mm_y) + 1e-9)
        confidence = float(np.clip(ratio, 0, 1))
    except Exception:
        return 0.1, 0.1, 0.0

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