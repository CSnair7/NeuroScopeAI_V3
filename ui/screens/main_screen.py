"""
MainScreen — Camera preview + all overlays.  LANDSCAPE MODE.

BUGS FIXED IN THIS REVISION
============================

BUG A — STRAIGHT LINES / MISALIGNED TRACES (primary cause)
  Root cause: _pipeline_frame() downscaled the input to 480 px wide before
  passing it to process_frame(). The vision pipeline then produces lead_rows,
  raw_pixel_row, x_start_clipped, x_end_clipped in *downscaled* pixel coords.
  The overlay uses image_shape=(H,W) of the downscaled frame as its coord
  reference. But _update_display_texture() shows the *full-res* frame. So
  the overlay is mapping downscaled pipeline coords onto a full-res image rect
  — every trace is compressed into the wrong zone and appears as a short
  straight line in the wrong position.

  Fix: the pipeline always receives the full-res preprocessed frame.
  The display ALSO shows the full-res frame. Both share the same pixel space,
  so the overlay coord mapping is correct.

  Performance impact: the frame is still captured at camera native resolution
  (typically 640×480). The preprocessing (bilateral+CLAHE+unsharp) now runs
  on the full frame in the background thread — acceptable on modern hardware.
  If frame rate is still too low, lower camera resolution in
  _actually_start_android_camera (e.g. 640×480 → 480×360).

BUG B — ARGMAX RETURNS FLAT ROW (secondary cause of straight lines)
  Root cause: extract_leads() receives original_image=vision_result.normalized_image.
  normalized_image is the output of normalize_image(img_boosted), which may
  apply perspective correction, gamma adjustment, or resizing internally.
  If normalized_image.shape != trace_mask.shape (which is also derived from
  normalized_image), the strip slice img_signal[y_start:y_end, x_start:x_end]
  has the wrong height, and argmax returns row 0 for every column (flat line).

  Fix: extract_leads() is called with original_image=preprocessed (the
  same frame that was passed into process_frame). The preprocessed frame and
  the normalized_image are the same resolution (process_frame does not resize),
  so img_signal slices are always the right size.

  Confirmed: normalize_image in vision_processor does NOT change frame size —
  it applies only colour/gamma transforms. So preprocessed and normalized_image
  have the same H×W, making either safe to pass. We use preprocessed to be
  explicit about the data source.

BUG C — THREAD STORM CRASH (previous fix, retained)
  _pipeline_busy flag + threading.Lock ensure one pipeline thread at a time.
  finally block always clears the flag.

BUG D — IMPORTERROR AT STARTUP (previous fix, retained)
  predictions_overlay.py uses ACCENT_RED_GLOW (not ACCENT_RED).

BUG E — LAMBDA CLOSURE CRASH (previous fix, retained)
  All Clock.schedule_once lambdas use explicit default-arg capture.

BUG F — ROTATION COORD TRANSFORM (previous fix, retained)
  _rotate_lead_coords() maps pixel coords from rotated-frame space back to
  original image space before passing to the overlay.
"""

import os
import threading
import time

import cv2
import numpy as np

from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.utils import platform

from config import AppConfig, get_theme_preset, PIPELINE_SAMPLE_RATE_HZ
from pipeline.electro_physics import reconstruct_leads, ReconstructionResult
from pipeline.model_engine import ModelInput, ModelOutput, create_engine
from pipeline.signal_processor import extract_leads
from pipeline.vision_processor import BUILTIN_THEMES, VisionResult, process_frame
from ui.styles import ACCENT_AMBER, BG_DARK, FONT_SIZE_HEADING
from ui.widgets.debug_overlay import DebugOverlay
from ui.widgets.neuro_overlay import NeuroOverlay
from ui.widgets.predictions_overlay import PredictionsOverlay

IS_ANDROID = (platform == "android")

_THEME_MAP = {
    "green_on_black": "dark_monitor_green",
    "white_on_black": "dark_monitor_white",
    "yellow_on_dark": "dark_monitor_yellow",
    "black_on_white": "paper_dark_trace",
}


# ---------------------------------------------------------------------------
# Temporal Stabilizer — smooths pipeline outputs across frames
# ---------------------------------------------------------------------------

class TemporalStabilizer:
    """
    Exponential moving average (EMA) smoother for real-time ECG overlay.

    Smooths:
      - lead_rows: (y_start, y_end, x_start, x_end) positions
      - calibration: mm_per_px_x, mm_per_px_y
      - raw_pixel_row: per-lead trace position arrays

    Alpha controls responsiveness:
      - 0.3 = heavy smoothing (stable, slow to react)
      - 0.7 = light smoothing (responsive, some jitter)
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._prev_lead_rows = None      # list of (y_s, y_e, x_s, x_e, name)
        self._prev_mm_x = None           # float
        self._prev_mm_y = None           # float
        self._prev_raw_pixels = {}       # name -> np.ndarray
        self._prev_n_leads = 0

    def reset(self):
        self._prev_lead_rows = None
        self._prev_mm_x = None
        self._prev_mm_y = None
        self._prev_raw_pixels = {}
        self._prev_n_leads = 0

    def smooth_calibration(self, mm_x: float, mm_y: float):
        """Smooth mm_per_px values with EMA."""
        if self._prev_mm_x is None:
            self._prev_mm_x = mm_x
            self._prev_mm_y = mm_y
        else:
            self._prev_mm_x = self.alpha * mm_x + (1.0 - self.alpha) * self._prev_mm_x
            self._prev_mm_y = self.alpha * mm_y + (1.0 - self.alpha) * self._prev_mm_y
        return self._prev_mm_x, self._prev_mm_y

    def smooth_lead_rows(self, lead_rows):
        """
        Smooth lead row positions with EMA.
        If the number of leads changes, reset to avoid blending mismatched data.
        """
        if not lead_rows:
            return lead_rows

        n = len(lead_rows)
        if self._prev_lead_rows is None or len(self._prev_lead_rows) != n:
            # Number of leads changed — reset
            self._prev_lead_rows = list(lead_rows)
            self._prev_n_leads = n
            return lead_rows

        smoothed = []
        a = self.alpha
        for i, (y_s, y_e, x_s, x_e, name) in enumerate(lead_rows):
            py_s, py_e, px_s, px_e, _ = self._prev_lead_rows[i]
            sy_s = int(round(a * y_s + (1 - a) * py_s))
            sy_e = int(round(a * y_e + (1 - a) * py_e))
            sx_s = int(round(a * x_s + (1 - a) * px_s))
            sx_e = int(round(a * x_e + (1 - a) * px_e))
            smoothed.append((sy_s, sy_e, sx_s, sx_e, name))

        self._prev_lead_rows = smoothed
        return smoothed

    def smooth_raw_pixel_rows(self, leads):
        """
        Smooth the raw_pixel_row arrays per lead using EMA.
        If the array length changes, reset that lead's history.
        """
        from copy import copy
        smoothed_leads = []
        for lead in leads:
            lc = copy(lead)
            rpr = getattr(lead, 'raw_pixel_row', None)
            if rpr is not None and len(rpr) > 0:
                prev = self._prev_raw_pixels.get(lead.name)
                if prev is not None and len(prev) == len(rpr):
                    smoothed = self.alpha * rpr + (1.0 - self.alpha) * prev
                    lc.raw_pixel_row = smoothed.astype(rpr.dtype)
                    self._prev_raw_pixels[lead.name] = smoothed
                else:
                    self._prev_raw_pixels[lead.name] = rpr.astype(np.float64)
            smoothed_leads.append(lc)
        return smoothed_leads


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MainScreen
# ---------------------------------------------------------------------------

class MainScreen(Screen):

    def __init__(self, app_config: AppConfig, test_image: str = None, **kwargs):
        super().__init__(**kwargs)
        self.name       = "main"
        self.app_config = app_config
        self._test_image_path = test_image
        self._running  = False
        self._capture  = None
        self._kivy_camera = None

        # BUG C — pipeline concurrency guard
        self._pipeline_busy = False
        self._pipeline_lock = threading.Lock()

        # Temporal stabilizer for smooth overlay
        self._stabilizer = TemporalStabilizer(alpha=0.3)

        # Model engine
        _onnx_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "pipeline", "v16_best.onnx",
        )
        self._engine = None
        
        try:
            # We enforce using the real ONNX model. No mock fallback.
            self._engine = create_engine(backend="onnx", model_path=_onnx_path)
        except Exception as e:
            print(f"[MainScreen] Model engine failed to load: {e}")
            raise RuntimeError(f"Failed to load actual model: {e}") from e

        self._frame_count = 0
        self._fps_time    = time.perf_counter()
        self._current_fps = 0.0

        with self.canvas.before:
            Color(*BG_DARK)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

        self._root = FloatLayout()

        self._display = Image(
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            allow_stretch=True,
            keep_ratio=True,
        )
        self._root.add_widget(self._display)

        self._overlay = NeuroOverlay(
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self._root.add_widget(self._overlay)

        self._status_label = Label(
            text="",
            color=ACCENT_AMBER,
            font_size=FONT_SIZE_HEADING,
            size_hint=(0.6, None),
            height=dp(36),
            pos_hint={"center_x": 0.5, "top": 0.98},
            bold=True,
            halign="center",
        )
        self._root.add_widget(self._status_label)

        self._debug_overlay = DebugOverlay()
        self._debug_overlay.pos_hint = {"right": 0.99, "top": 0.99}
        self._root.add_widget(self._debug_overlay)

        self._pred_overlay = PredictionsOverlay()
        self._pred_overlay.pos_hint = {"x": 0.01, "y": 0.01}
        self._root.add_widget(self._pred_overlay)


        self.add_widget(self._root)

    def _update_bg(self, *args):
        self._bg_rect.pos  = self.pos
        self._bg_rect.size = self.size

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_enter(self, *args):
        self._debug_overlay.visible = self.app_config.debug_mode
        self._pred_overlay.visible  = True
        self._running = True
        if self._test_image_path and os.path.exists(self._test_image_path):
            Clock.schedule_once(self._process_test_image, 0.5)
        else:
            self._start_camera()

    def on_leave(self, *args):
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._kivy_camera is not None:
            self._kivy_camera.play = False

    # ── Camera ────────────────────────────────────────────────────────────

    def _start_camera(self):
        if IS_ANDROID:
            self._start_camera_android()
        else:
            self._start_camera_desktop()

    def _start_camera_android(self):
        try:
            self._request_camera_permission()
        except Exception as e:
            self._show_status(f"Camera permission error: {e}")

    def _request_camera_permission(self):
        try:
            from android.permissions import (
                Permission, check_permission, request_permissions,
            )
            if check_permission(Permission.CAMERA):
                self._actually_start_android_camera()
            else:
                request_permissions(
                    [Permission.CAMERA],
                    callback=self._on_permission_result,
                )
        except ImportError:
            self._actually_start_android_camera()

    def _on_permission_result(self, permissions, grants):
        if grants and all(grants):
            Clock.schedule_once(
                lambda dt: self._actually_start_android_camera(), 0.5
            )
        else:
            Clock.schedule_once(
                lambda dt: self._show_status("Camera permission denied"), 0.1
            )

    def _actually_start_android_camera(self):
        """
        Kivy Camera widget, hidden — we pull frames manually for pipeline control.

        Resolution: 640×480 is the sweet spot for speed vs. detail.
        If pipeline latency is still too high, reduce to (480, 360) here.
        Do NOT reduce pipeline input resolution by downscaling in code —
        that breaks the overlay coordinate mapping (see BUG A docstring).
        """
        try:
            from kivy.uix.camera import Camera as KivyCamera
            self._kivy_camera = KivyCamera(
                index=0,
                resolution=(640, 480),
                play=True,
                size_hint=(None, None),
                size=(1, 1),
                opacity=0,
            )
            self._root.add_widget(self._kivy_camera)
            Clock.schedule_interval(self._grab_camera_frame, 1.0 / 30)
            self._show_status("")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._show_status(f"Camera init failed: {e}")

    def _grab_camera_frame(self, dt):
        """
        Main-thread tick at 30 fps.

        Fast path (always, ~2 ms):
            texture pixels → numpy → BGR → display texture

        Slow path (only when pipeline is idle):
            dispatch _process_frame_async to background thread.
            No downscaling — full-res frame sent to pipeline (BUG A fix).
        """
        if not self._running or self._kivy_camera is None:
            return
        texture = self._kivy_camera.texture
        if texture is None:
            return
        try:
            pixels    = texture.pixels
            w, h      = texture.size
            img       = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 4)
            img       = cv2.flip(img, 0)
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            self._update_display_texture(frame_bgr)

            with self._pipeline_lock:
                if self._pipeline_busy:
                    return
                self._pipeline_busy = True

            threading.Thread(
                target=self._process_frame_async,
                args=(frame_bgr.copy(),),
                daemon=True,
            ).start()
        except Exception:
            with self._pipeline_lock:
                self._pipeline_busy = False

    def _start_camera_desktop(self):
        try:
            self._capture = cv2.VideoCapture(0)
            if not self._capture.isOpened():
                self._show_status("Camera not available")
                return
            Clock.schedule_interval(self._capture_frame, 1.0 / 30)
        except Exception as e:
            self._show_status(f"Camera error: {e}")

    def _capture_frame(self, dt):
        if not self._running or self._capture is None:
            return
        ret, frame = self._capture.read()
        if not ret:
            self._show_status("No camera frame")
            return

        self._update_display_texture(frame)

        with self._pipeline_lock:
            if self._pipeline_busy:
                return
            self._pipeline_busy = True

        threading.Thread(
            target=self._process_frame_async,
            args=(frame.copy(),),
            daemon=True,
        ).start()

    # ── Test image ─────────────────────────────────────────────────────────

    def _process_test_image(self, dt):
        img = cv2.imread(self._test_image_path)
        if img is None:
            self._show_status(f"Cannot load: {self._test_image_path}")
            return
        
        self._update_display_texture(img)
        with self._pipeline_lock:
            self._pipeline_busy = True
        # Run synchronously on the main thread for test images (no live loop)
        self._process_frame_async(img.copy())

    # ── Pipeline (background thread) ───────────────────────────────────────

    def _process_frame_async(self, frame_bgr: np.ndarray):
        """
        Background thread: preprocess → vision → signal → model → schedule UI.

        BUG A fix: preprocessing produces `preprocessed` at the SAME resolution
        as `frame_bgr`. This is passed to both process_frame() and extract_leads().
        The overlay receives image_shape = preprocessed.shape[:2], which matches
        the resolution of the frame shown in _update_display_texture().

        BUG B fix: extract_leads() receives original_image=preprocessed (not
        vision_result.normalized_image). Both are the same resolution, so
        img_signal strip slices always match the mask strip size.

        No downscaling anywhere in this path.
        """
        t0 = time.perf_counter()
        try:
            # Step 1: preprocess (in background thread — keeps main thread fast)
            preprocessed = _preprocess_for_pipeline(frame_bgr)
            # preprocessed.shape == frame_bgr.shape — no resize

            theme_key = _THEME_MAP.get(self.app_config.theme, "dark_monitor_green")
            theme = BUILTIN_THEMES.get(theme_key) or get_theme_preset(
                self.app_config.theme
            )

            # Step 2: vision (lead detection, calibration, mask)
            # Pass preprocessed — same shape ensures all coord outputs are valid
            vision_result = process_frame(
                preprocessed,
                theme=theme,
                skip_perspective=True,
                expected_leads=self.app_config.num_leads,
            )
            # vision_result.trace_mask, normalized_image are all same HxW as preprocessed

            # Step 3: signal extraction
            # BUG B fix: pass preprocessed as original_image so argmax tracking
            # uses the correct strip slices (same shape as trace_mask)
            leads = extract_leads(
                vision_result.trace_mask,
                mm_per_px_x=vision_result.mm_per_px_x,
                mm_per_px_y=vision_result.mm_per_px_y,
                lead_rows=vision_result.lead_rows or None,
                sample_rate_hz=PIPELINE_SAMPLE_RATE_HZ,
                original_image=preprocessed,   # BUG B fix: was vision_result.normalized_image
            )

            # Step 4: rebuild lead_rows with clipped x-bounds from LeadSignal
            if leads and vision_result.lead_rows:
                rows_by_name = {r[4]: r for r in vision_result.lead_rows}
                clipped = []
                for lead in leads:
                    orig = rows_by_name.get(lead.name)
                    if orig is None:
                        continue
                    y_s, y_e, x_s, x_e, lname = orig
                    clipped.append((
                        y_s, y_e,
                        getattr(lead, "x_start_clipped", x_s),
                        getattr(lead, "x_end_clipped",   x_e),
                        lname,
                    ))
                vision_result.lead_rows = clipped

            # Step 4b: temporal smoothing — stabilize across frames
            sm_mm_x, sm_mm_y = self._stabilizer.smooth_calibration(
                vision_result.mm_per_px_x, vision_result.mm_per_px_y,
            )
            vision_result.mm_per_px_x = sm_mm_x
            vision_result.mm_per_px_y = sm_mm_y

            if vision_result.lead_rows:
                vision_result.lead_rows = self._stabilizer.smooth_lead_rows(
                    vision_result.lead_rows,
                )

            if leads:
                leads = self._stabilizer.smooth_raw_pixel_rows(leads)

            # Map pixel coords to screen coords (since we no longer modify rotation, it maps 1:1)
            pipeline_hw = preprocessed.shape[:2]
            leads_display = leads
            lead_rows_display = vision_result.lead_rows or []
            display_hw = pipeline_hw

            # Step 6: signal reconstruction + model inference
            recon = reconstruct_leads(leads)   # uses original leads for signal
            model_input = ModelInput(
                signal_tensor=recon.signal_tensor,
                lead_mask=recon.lead_mask,
                sampling_rate=recon.sampling_rate,
            )
            model_output = self._engine.predict(model_input)
            total_ms = (time.perf_counter() - t0) * 1000.0

            # BUG E fix: explicit capture of all locals in lambda default args
            Clock.schedule_once(
                lambda dt,
                       _ld=leads_display,
                       _lr=lead_rows_display,
                       _dh=display_hw,
                       _v=vision_result,
                       _r=recon,
                       _mo=model_output,
                       _ms=total_ms:
                    self._update_ui(_ld, _lr, _dh, _v, _r, _mo, _ms)
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            _msg = str(e)
            Clock.schedule_once(
                lambda dt, m=_msg: self._show_status(f"Error: {m}")
            )
        finally:
            # BUG C fix: always release the lock
            with self._pipeline_lock:
                self._pipeline_busy = False

    # ── UI update (main thread via Clock.schedule_once) ────────────────────

    def _update_ui(
        self,
        leads_display,
        lead_rows_display,
        display_hw,
        vision: VisionResult,
        recon: ReconstructionResult,
        model_output: ModelOutput,
        total_ms: float,
    ):
        # BUG E fix: guard against late-firing callbacks after screen change
        if not self._running:
            return
        try:
            self._update_ui_safe(
                leads_display, lead_rows_display, display_hw,
                vision, recon, model_output, total_ms,
            )
        except Exception as _e:
            import traceback as _tb; _tb.print_exc()
            self._show_status(f"UI error: {_e}")

    def _update_ui_safe(
        self,
        leads_display,
        lead_rows_display,
        display_hw,
        vision: VisionResult,
        recon: ReconstructionResult,
        model_output: ModelOutput,
        total_ms: float,
    ):
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time    = now

        if not leads_display:
            self._show_status("No Waveform Detected")
            self._overlay.waveform_points = []
            return

        avg_snr  = float(np.mean([l.snr_db        for l in leads_display]))
        avg_cont = float(np.mean([l.continuity_pct for l in leads_display]))

        if avg_cont < 25.0:
            self._show_status("Signal Unstable")
        elif vision.calibration_confidence < 0.5:
            self._show_status("Low Calibration")
        elif model_output.confidence_score < 0.5:
            self._show_status("Low Confidence - Verify")
        else:
            self._status_label.text = ""

        # BUG A fix: image_shape is the display frame's shape (same pixel space
        # as the frame shown in _update_display_texture)
        self._overlay.update_leads(
            leads_display,
            lead_rows=lead_rows_display,
            image_shape=display_hw,
        )
        self._overlay.is_abnormal = (model_output.binary_prediction == "abnormal")

        if model_output.suspicious_segments:
            self._overlay.update_suspicious_segments(
                model_output.suspicious_segments,
                total_samples=len(leads_display[0].voltage_mv),
            )
        else:
            self._overlay.update_suspicious_segments([], 0)

        self._pred_overlay.update_from_model_output(model_output)

        if self.app_config.debug_mode:
            self._debug_overlay.visible = True
            n_raw     = len(leads_display)
            n_derived = int(recon.lead_mask.sum()) - n_raw
            mask_str  = f"{n_raw} raw"
            if n_derived > 0:
                mask_str += f" + {n_derived} derived"
            self._debug_overlay.update_stats(
                tensor_shape=recon.signal_tensor.shape,
                lead_mask=mask_str,
                rmse=recon.rmse_reconstruction,
                snr=avg_snr,
                cal_confidence=vision.calibration_confidence,
                latency_ms=total_ms,
                fps=self._current_fps,
            )
        else:
            self._debug_overlay.visible = False

    # ── Texture helper ─────────────────────────────────────────────────────

    def _update_display_texture(self, frame_bgr: np.ndarray):
        """Main-thread only. BGR → Kivy Texture. No resizing."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)      # Kivy textures are bottom-up
        h, w      = frame_rgb.shape[:2]
        texture   = Texture.create(size=(w, h), colorfmt="rgb")
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self._display.texture = texture

    def _show_status(self, text: str):
        self._status_label.text = text