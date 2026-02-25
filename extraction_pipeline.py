"""
ECG Signal Extraction Pipeline
Extracts ECG waveforms from phone camera images of ECG printouts.

Dependencies:
    pip install opencv-python numpy scipy scikit-image matplotlib
"""

import cv2
import numpy as np
from scipy import signal, ndimage
from skimage import morphology, filters
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    mm_per_px_x: float
    mm_per_px_y: float
    affine_transform: np.ndarray        # 2x3 affine matrix
    confidence: float                   # 0–1


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
    rejection_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — Image normalization
# ---------------------------------------------------------------------------

def normalize_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Correct perspective, even out illumination, and compensate for lens distortion.
    Returns a normalized BGR image.
    """
    log.info("Step 1: Normalizing image")

    # --- 1a. Lens distortion correction (placeholder camera matrix) ---
    # In production, calibrate once per device model and store intrinsics.
    h, w = img_bgr.shape[:2]
    fx = fy = max(h, w) * 0.9          # rough focal length estimate
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    # Mild barrel distortion typical of phone cameras
    dist_coeffs = np.array([-0.05, 0.01, 0, 0, 0], dtype=np.float32)
    img_undist = cv2.undistort(img_bgr, K, dist_coeffs)

    # --- 1b. Perspective correction ---
    img_persp = _correct_perspective(img_undist)

    # --- 1c. CLAHE on L-channel of LAB for even illumination ---
    lab = cv2.cvtColor(img_persp, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    log.info("  Image normalized: %dx%d", img_norm.shape[1], img_norm.shape[0])
    return img_norm


def _correct_perspective(img: np.ndarray) -> np.ndarray:
    """Detect paper corners and apply homographic warp."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.warning("  No contours found; skipping perspective correction")
        return img

    # Find largest quadrilateral contour (the paper boundary)
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        log.warning("  Could not isolate paper boundary; skipping perspective correction")
        return img

    pts_src = _order_points(approx.reshape(4, 2).astype(np.float32))
    h, w = img.shape[:2]
    pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img, M, (w, h))


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ---------------------------------------------------------------------------
# Step 2 — Trace isolation
# ---------------------------------------------------------------------------

def isolate_trace(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segment the ECG trace from the grid/background using adaptive LAB thresholding.
    Returns a binary mask (uint8, 0 or 255) where 255 = trace.
    """
    log.info("Step 2: Isolating ECG trace")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Unsharp mask on L-channel to sharpen trace edges before thresholding
    l_channel = lab[:, :, 0].astype(np.float32)
    l_blurred = cv2.GaussianBlur(l_channel, (0, 0), 1.0)
    l_sharp = cv2.addWeighted(l_channel, 1.8, l_blurred, -0.8, 0)
    lab[:, :, 0] = np.clip(l_sharp, 0, 255).astype(np.uint8)

    # Detect whether trace is chromatic (color) or achromatic (black)
    a_std = float(np.std(lab[:, :, 1]))
    b_std = float(np.std(lab[:, :, 2]))
    is_chromatic = (a_std > 8.0 or b_std > 8.0)

    if is_chromatic:
        log.info("  Chromatic trace detected — using k-means segmentation")
        mask = _kmeans_segment(lab)
    else:
        log.info("  Achromatic trace detected — using adaptive Otsu thresholding")
        mask = _adaptive_otsu_segment(lab[:, :, 0])

    log.info("  Trace coverage: %.1f%%", 100 * mask.sum() / mask.size)
    return mask


def _kmeans_segment(lab: np.ndarray) -> np.ndarray:
    """k-means (k=3) on LAB pixels; return mask of cluster closest to trace color."""
    h, w = lab.shape[:2]
    # Downsample for speed
    scale = 0.25
    small = cv2.resize(lab, (int(w * scale), int(h * scale)))
    pixels = small.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Assume trace cluster is darkest (lowest L) and most chromatic
    scores = -centers[:, 0] + np.sqrt(centers[:, 1] ** 2 + centers[:, 2] ** 2)
    trace_cluster = int(np.argmax(scores))

    # Apply labels to full-resolution image
    pixels_full = lab.reshape(-1, 3).astype(np.float32)
    _, labels_full, _ = cv2.kmeans(pixels_full, 3, None, criteria, 3,
                                   cv2.KMEANS_USE_INITIAL_LABELS,
                                   bestLabels=np.tile(labels, (int(1 / scale ** 2) + 1, 1)
                                                      ).flatten()[:h * w].reshape(-1, 1))
    mask = (labels_full.flatten() == trace_cluster).reshape(h, w).astype(np.uint8) * 255
    return mask


def _adaptive_otsu_segment(gray: np.ndarray) -> np.ndarray:
    """Tile-based Otsu thresholding to handle uneven illumination."""
    tile_h, tile_w = gray.shape[0] // 4, gray.shape[1] // 4
    mask = np.zeros_like(gray, dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            tile = gray[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
            _, t = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = t
    return mask


# ---------------------------------------------------------------------------
# Step 3 — Morphological cleanup
# ---------------------------------------------------------------------------

def clean_trace(mask: np.ndarray, mm_per_px: float = 0.1) -> np.ndarray:
    """
    Remove grid residuals with erosion, reconstruct trace with directional dilation.
    mm_per_px: estimated ratio used to size kernels dynamically.
    """
    log.info("Step 3: Morphological cleanup")

    # Pre-blur to smooth noise before morphological ops
    blurred = cv2.GaussianBlur(mask, (0, 0), 1.2)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Dynamic erosion kernel based on spatial calibration
    ek = max(3, round(0.3 / mm_per_px))
    ek = ek if ek % 2 == 1 else ek + 1         # ensure odd
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ek, ek))
    eroded = cv2.erode(binary, erode_kernel, iterations=1)

    # Directional (horizontal) dilation to rebuild continuity along signal axis
    dk = max(7, round(0.7 / mm_per_px))
    dk = dk if dk % 2 == 1 else dk + 1
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dk, 1))
    dilated = cv2.dilate(eroded, dilate_kernel, iterations=1)

    # Remove connected components too small or too vertically oriented to be traces
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    min_area = 0.005 * mask.size
    clean = np.zeros_like(dilated)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w / (h_comp + 1e-6)
        if area >= min_area and aspect > 2.0:   # traces are wide, not tall
            clean[labels == i] = 255

    log.info("  Retained %d components after filtering", int(clean.max() > 0))
    return clean


# ---------------------------------------------------------------------------
# Step 4 — Grid calibration
# ---------------------------------------------------------------------------

def calibrate_grid(img_gray: np.ndarray) -> CalibrationResult:
    """
    Detect grid lines via Hough transform, compute per-axis mm/px ratios,
    and return a 2D affine calibration transform.
    ECG standard: small square = 1mm, large square = 5mm.
    """
    log.info("Step 4: Calibrating grid")

    edges = cv2.Canny(img_gray, 30, 80)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h_space, angles, dists = hough_line(edges, theta=tested_angles)
    _, angles_peak, dists_peak = hough_line_peaks(h_space, angles, dists,
                                                   min_distance=5, threshold=0.3 * h_space.max())

    # Separate near-horizontal and near-vertical lines
    horiz = [(a, d) for a, d in zip(angles_peak, dists_peak) if abs(a) < np.radians(10)]
    vert  = [(a, d) for a, d in zip(angles_peak, dists_peak) if abs(a - np.pi / 2) < np.radians(10)]

    mm_per_px_x = _infer_mm_per_px([d for _, d in vert])
    mm_per_px_y = _infer_mm_per_px([d for _, d in horiz])

    log.info("  Calibration — X: %.4f mm/px, Y: %.4f mm/px", mm_per_px_x, mm_per_px_y)

    # Build affine transform (scale only; rotation already handled in step 1)
    affine = np.array([[mm_per_px_x, 0, 0],
                       [0, mm_per_px_y, 0]], dtype=np.float64)

    # Confidence: penalise if X and Y ratios disagree significantly
    ratio = min(mm_per_px_x, mm_per_px_y) / max(mm_per_px_x, mm_per_px_y)
    confidence = float(np.clip(ratio, 0, 1))

    return CalibrationResult(mm_per_px_x=mm_per_px_x,
                             mm_per_px_y=mm_per_px_y,
                             affine_transform=affine,
                             confidence=confidence)


def _infer_mm_per_px(positions: list[float]) -> float:
    """
    Given a list of parallel line positions (pixels), infer mm/px by matching
    spacings to ECG grid periodicities (1mm and 5mm small/large squares).
    Falls back to median spacing if matching is ambiguous.
    """
    if len(positions) < 2:
        log.warning("  Too few grid lines detected; using fallback mm/px=0.1")
        return 0.1

    positions = sorted(positions)
    spacings = np.diff(positions)
    median_spacing = float(np.median(spacings))

    # Try to match to 1mm or 5mm ECG squares
    for grid_mm in [5.0, 1.0]:
        ratio = grid_mm / median_spacing
        # Accept if ratio is plausible for phone capture (0.01 – 1.0 mm/px)
        if 0.01 <= ratio <= 1.0:
            return ratio

    # Last resort
    return 0.1


# ---------------------------------------------------------------------------
# Step 5 — Pixel-to-voltage conversion
# ---------------------------------------------------------------------------

def extract_lead_signals(
    mask: np.ndarray,
    calibration: CalibrationResult,
    lead_rows: Optional[list[tuple[int, int, str]]] = None,
    sample_rate_hz: float = 500.0,
) -> list[LeadSignal]:
    """
    Convert mask pixel columns to voltage traces per lead.

    lead_rows: list of (y_start, y_end, lead_name) for each row in a multi-lead strip.
               If None, treats entire image as single lead.
    """
    log.info("Step 5: Pixel-to-voltage conversion")

    h, w = mask.shape
    if lead_rows is None:
        lead_rows = [(0, h, "Lead_I")]

    leads = []
    for y_start, y_end, name in lead_rows:
        row_mask = mask[y_start:y_end, :]
        row_h = y_end - y_start

        # Find topmost trace pixel in each column (inverted: lower y = higher voltage)
        voltage_px = np.full(w, np.nan)
        for col in range(w):
            col_data = row_mask[:, col]
            trace_rows = np.where(col_data > 0)[0]
            if len(trace_rows):
                voltage_px[col] = float(trace_rows[0])  # topmost pixel

        # Interpolate gaps (NaN = missing trace)
        nan_mask = np.isnan(voltage_px)
        continuity_pct = 100.0 * (1 - nan_mask.sum() / w)
        if nan_mask.all():
            log.warning("  %s: entirely missing — skipping", name)
            continue
        idx = np.arange(w)
        voltage_px = np.interp(idx, idx[~nan_mask], voltage_px[~nan_mask])

        # Rolling median baseline to track drift (window = 200ms)
        window = max(3, int(0.2 * sample_rate_hz))
        baseline_px = _rolling_median(voltage_px, window)

        # Convert to mV:  Voltage = (y_px - baseline_px) * mm/px_y * 0.1 mV/mm
        # Negate because image y increases downward
        voltage_mv = -(voltage_px - baseline_px) * calibration.mm_per_px_y * 0.1
        baseline_drift_mv = -(baseline_px - np.median(baseline_px)) * calibration.mm_per_px_y * 0.1

        time_s = np.arange(w) / (sample_rate_hz)

        snr = _compute_snr(voltage_mv)
        leads.append(LeadSignal(name=name,
                                voltage_mv=voltage_mv,
                                time_s=time_s,
                                snr_db=snr,
                                continuity_pct=continuity_pct,
                                baseline_drift_mv=baseline_drift_mv))
        log.info("  %s — SNR: %.1f dB, continuity: %.1f%%", name, snr, continuity_pct)

    return leads


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling median via 1D uniform filter approximation."""
    # scipy median_filter is exact; use it for correctness
    return ndimage.median_filter(x, size=window, mode="nearest")


def _compute_snr(voltage_mv: np.ndarray) -> float:
    """SNR = 20*log10(P_signal / P_noise) on QRS-gated windows."""
    # Broad QRS detection: peaks above 0.3× max amplitude
    abs_v = np.abs(voltage_mv)
    threshold = 0.3 * abs_v.max()
    signal_mask = abs_v > threshold
    noise_mask = ~signal_mask

    p_signal = float(np.mean(voltage_mv[signal_mask] ** 2)) if signal_mask.any() else 1e-9
    p_noise  = float(np.mean(voltage_mv[noise_mask]  ** 2)) if noise_mask.any()  else 1e-12

    if p_noise == 0:
        return 99.0
    return float(20 * np.log10(np.sqrt(p_signal) / (np.sqrt(p_noise) + 1e-12)))


# ---------------------------------------------------------------------------
# Step 6 — Quality control
# ---------------------------------------------------------------------------

SNR_THRESHOLD_DB       = 15.0
CONTINUITY_THRESHOLD   = 95.0
QRS_AMP_RANGE_MV       = (0.1, 3.5)
QRS_WIDTH_RANGE_MS     = (40, 120)
RR_CV_THRESHOLD        = 0.30


def quality_control(leads: list[LeadSignal],
                    calibration: CalibrationResult,
                    sample_rate_hz: float = 500.0) -> tuple[bool, list[str]]:
    """
    Run all QC gates. Returns (passed, list_of_rejection_reasons).
    Per-lead failures are reported individually so partial acceptance is possible.
    """
    log.info("Step 6: Quality control")
    reasons = []

    if not leads:
        reasons.append("NO_LEADS: No lead signals could be extracted")
        return False, reasons

    # Calibration confidence
    if calibration.confidence < 0.8:
        reasons.append(f"CALIBRATION_ASYMMETRY: X/Y scale ratio {calibration.confidence:.2f} < 0.80 "
                       "(possible uncorrected perspective distortion)")

    for lead in leads:

        # SNR gate
        if lead.snr_db < SNR_THRESHOLD_DB:
            reasons.append(f"{lead.name} SNR_FAIL: {lead.snr_db:.1f} dB < {SNR_THRESHOLD_DB} dB")

        # Trace continuity gate
        if lead.continuity_pct < CONTINUITY_THRESHOLD:
            reasons.append(f"{lead.name} CONTINUITY_FAIL: {lead.continuity_pct:.1f}% < {CONTINUITY_THRESHOLD}%")

        # Morphological plausibility: R-peak amplitude
        r_peaks, _ = signal.find_peaks(lead.voltage_mv,
                                        height=QRS_AMP_RANGE_MV[0],
                                        distance=int(0.3 * sample_rate_hz))
        if len(r_peaks) == 0:
            reasons.append(f"{lead.name} NO_QRS: No R-peaks detected above {QRS_AMP_RANGE_MV[0]} mV")
            continue

        peak_amps = lead.voltage_mv[r_peaks]
        if np.any(peak_amps > QRS_AMP_RANGE_MV[1]):
            reasons.append(f"{lead.name} AMP_OVERFLOW: Peak {peak_amps.max():.2f} mV > {QRS_AMP_RANGE_MV[1]} mV "
                           "(possible calibration error)")

        # RR interval regularity
        if len(r_peaks) >= 3:
            rr_intervals = np.diff(r_peaks) / sample_rate_hz
            rr_cv = float(np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-9))
            if rr_cv > RR_CV_THRESHOLD:
                reasons.append(f"{lead.name} IRREGULAR_RR: CV={rr_cv:.2f} > {RR_CV_THRESHOLD} "
                               "(possible arrhythmia or motion artifact)")

        # Motion artifact proxy: power in 25–40 Hz band
        freqs, psd = signal.welch(lead.voltage_mv, fs=sample_rate_hz, nperseg=min(256, len(lead.voltage_mv)))
        hf_band = (freqs >= 25) & (freqs <= 40)
        total_band = freqs <= sample_rate_hz / 2
        hf_ratio = float(psd[hf_band].sum() / (psd[total_band].sum() + 1e-12))
        if hf_ratio > 0.15:
            reasons.append(f"{lead.name} MOTION_ARTIFACT: HF power ratio {hf_ratio:.2f} > 0.15")

        # Polarity check (Lead I: QRS should be net positive in normal sinus)
        if lead.name in ("Lead_I", "I"):
            net_deflection = float(lead.voltage_mv[r_peaks].mean())
            if net_deflection < 0:
                reasons.append(f"{lead.name} POLARITY_INVERTED: Mean QRS = {net_deflection:.2f} mV "
                               "(possible lead reversal or baseline misdetection)")

    passed = len(reasons) == 0
    if passed:
        log.info("  QC passed for all leads")
    else:
        log.warning("  QC issues:\n    " + "\n    ".join(reasons))

    return passed, reasons


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: str,
    lead_rows: Optional[list[tuple[int, int, str]]] = None,
    sample_rate_hz: float = 500.0,
    plot: bool = True,
) -> ExtractionResult:
    """
    Full ECG extraction pipeline.

    Args:
        image_path:     Path to phone camera image of ECG printout.
        lead_rows:      [(y_start, y_end, 'Lead_name'), ...] for multi-lead strips.
                        Pass None for single-lead images.
        sample_rate_hz: Assumed digitisation rate (pixels/sec mapping).
        plot:           Whether to display a results plot.

    Returns:
        ExtractionResult with per-lead signals and QC status.
    """
    # Load
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Step 1
    img_norm = normalize_image(img_bgr)

    # Step 2
    trace_mask = isolate_trace(img_norm)

    # Step 3 — use rough default mm/px; will be refined after calibration
    clean_mask = clean_trace(trace_mask, mm_per_px=0.1)

    # Step 4
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    calibration = calibrate_grid(gray)

    # Re-clean with calibrated mm/px
    clean_mask = clean_trace(trace_mask, mm_per_px=calibration.mm_per_px_y)

    # Step 5
    leads = extract_lead_signals(clean_mask, calibration, lead_rows, sample_rate_hz)

    # Step 6
    passed, reasons = quality_control(leads, calibration, sample_rate_hz)

    result = ExtractionResult(leads=leads,
                              calibration=calibration,
                              passed_qc=passed,
                              rejection_reasons=reasons)

    if plot:
        _plot_results(img_norm, clean_mask, result)

    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_results(img: np.ndarray, mask: np.ndarray, result: ExtractionResult) -> None:
    n_leads = len(result.leads)
    if n_leads == 0:
        return

    fig, axes = plt.subplots(2 + n_leads, 1, figsize=(14, 4 + 3 * n_leads))
    fig.suptitle(f"ECG Extraction  |  QC: {'PASS' if result.passed_qc else 'FAIL'}", fontsize=13)

    # Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Normalized Input Image")
    axes[0].axis("off")

    # Cleaned mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Extracted Trace Mask")
    axes[1].axis("off")

    # Per-lead voltages
    for i, lead in enumerate(result.leads):
        ax = axes[2 + i]
        ax.plot(lead.time_s, lead.voltage_mv, color="#1a73e8", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("mV")
        ax.set_title(f"{lead.name}  |  SNR={lead.snr_db:.1f} dB  |  "
                     f"Continuity={lead.continuity_pct:.1f}%")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "ecg_sample.jpg"

    # Example multi-lead configuration for a standard 3-row strip
    # Adjust y_start/y_end to match the actual image layout
    example_leads = [
        (50,  250, "Lead_I"),
        (260, 460, "Lead_II"),
        (470, 670, "Lead_III"),
    ]

    result = run_pipeline(
        image_path=image_path,
        lead_rows=example_leads,
        sample_rate_hz=500.0,
        plot=True,
    )

    print("\n=== Extraction Summary ===")
    print(f"QC Status : {'PASSED' if result.passed_qc else 'FAILED'}")
    print(f"Calibration: X={result.calibration.mm_per_px_x:.4f} mm/px, "
          f"Y={result.calibration.mm_per_px_y:.4f} mm/px "
          f"(confidence={result.calibration.confidence:.2f})")
    for lead in result.leads:
        print(f"  {lead.name}: SNR={lead.snr_db:.1f} dB, "
              f"continuity={lead.continuity_pct:.1f}%, "
              f"peak_amp={lead.voltage_mv.max():.2f} mV")
    if result.rejection_reasons:
        print("\nRejection Reasons:")
        for r in result.rejection_reasons:
            print(f"  - {r}")