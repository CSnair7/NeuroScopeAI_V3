"""
MainScreen — Camera preview + all overlays.

Hosts: NeuroOverlay, DiagnosticPanel, DebugOverlay.
Runs the pipeline on a background thread and updates the UI via Clock.
"""

import os
import sys
import threading
import time
import numpy as np
import cv2

from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.properties import ObjectProperty, BooleanProperty

from config import AppConfig, get_theme_preset, PIPELINE_SAMPLE_RATE_HZ
from pipeline.vision_processor import process_frame, VisionResult
from pipeline.signal_processor import extract_leads, LeadSignal
from pipeline.electro_physics import reconstruct_leads, ReconstructionResult
from pipeline.model_engine import (
    ModelInput, ModelOutput, MockModelEngine, create_engine,
    get_segment_explanation,
)
from ui.widgets.neuro_overlay import NeuroOverlay
from ui.widgets.diagnostic_panel import DiagnosticPanel
from ui.widgets.debug_overlay import DebugOverlay
from ui.styles import (
    BG_DARK, TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_AMBER,
    FONT_SIZE_HEADING, FONT_SIZE_BODY,
)


class MainScreen(Screen):
    """
    Primary analysis screen.

    Modes:
      - Camera mode: captures frames from webcam/phone camera
      - Test image mode: loads a static image for pipeline testing
    """

    def __init__(self, app_config: AppConfig, test_image: str = None, **kwargs):
        super().__init__(**kwargs)
        self.name = "main"
        self.app_config = app_config
        self._test_image_path = test_image
        self._running = False
        self._capture = None
        self._engine = create_engine(backend="mock", seed=42)
        self._frame_count = 0
        self._fps_time = time.perf_counter()
        self._current_fps = 0.0

        # Dark background
        with self.canvas.before:
            Color(*BG_DARK)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

        # Root float layout
        self._root = FloatLayout()

        # Camera/Image display
        self._display = Image(
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            allow_stretch=True,
            keep_ratio=True,
        )
        self._root.add_widget(self._display)

        # Waveform overlay
        self._overlay = NeuroOverlay(
            size_hint=(1, 0.35),
            pos_hint={"center_x": 0.5, "y": 0.35},
        )
        self._root.add_widget(self._overlay)

        # Diagnostic panel
        self._diag_panel = DiagnosticPanel()
        self._root.add_widget(self._diag_panel)

        # Debug overlay
        self._debug_overlay = DebugOverlay()
        self._root.add_widget(self._debug_overlay)

        # Status overlay (for edge cases)
        self._status_label = Label(
            text="",
            color=ACCENT_AMBER,
            font_size=FONT_SIZE_HEADING,
            size_hint=(1, None),
            height=40,
            pos_hint={"center_x": 0.5, "top": 0.95},
            bold=True,
        )
        self._root.add_widget(self._status_label)

        self.add_widget(self._root)

    def _update_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size

    def on_enter(self, *args):
        """Called when the screen becomes active."""
        self._debug_overlay.visible = self.app_config.debug_mode
        self._running = True

        if self._test_image_path and os.path.exists(self._test_image_path):
            # Test image mode — process once
            Clock.schedule_once(self._process_test_image, 0.5)
        else:
            # Camera mode — start capture loop
            self._start_camera()

    def on_leave(self, *args):
        """Called when leaving the screen."""
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # ------------------------------------------------------------------
    # Camera mode
    # ------------------------------------------------------------------

    def _start_camera(self):
        """Open webcam and schedule frame capture."""
        try:
            self._capture = cv2.VideoCapture(0)
            if not self._capture.isOpened():
                self._show_status("Camera not available")
                return
            Clock.schedule_interval(self._capture_frame, 1.0 / 30)
        except Exception as e:
            self._show_status(f"Camera error: {e}")

    def _capture_frame(self, dt):
        """Grab a frame and process it in a background thread."""
        if not self._running or self._capture is None:
            return

        ret, frame = self._capture.read()
        if not ret:
            self._show_status("No camera frame")
            return

        # Run pipeline in background to avoid blocking UI
        threading.Thread(
            target=self._process_frame_async,
            args=(frame.copy(),),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Test image mode
    # ------------------------------------------------------------------

    def _process_test_image(self, dt):
        """Load and process a static test image."""
        img = cv2.imread(self._test_image_path)
        if img is None:
            self._show_status(f"Cannot load: {self._test_image_path}")
            return

        self._process_frame_async(img)

    # ------------------------------------------------------------------
    # Pipeline processing (runs on background thread)
    # ------------------------------------------------------------------

    def _process_frame_async(self, frame_bgr: np.ndarray):
        """
        Full pipeline pass on a single frame.
        Results are dispatched to the UI thread via Clock.schedule_once.
        """
        t0 = time.perf_counter()

        try:
            theme = get_theme_preset(self.app_config.theme)

            # Module A: Vision
            vision_result = process_frame(
                frame_bgr, theme, skip_perspective=True,
                expected_leads=self.app_config.num_leads,
            )

            # Module B: Signal Processing
            leads = extract_leads(
                vision_result.trace_mask,
                mm_per_px_x=vision_result.mm_per_px_x,
                mm_per_px_y=vision_result.mm_per_px_y,
                lead_rows=vision_result.lead_rows or None,
                sample_rate_hz=PIPELINE_SAMPLE_RATE_HZ,
            )

            # Module C: Electro-Physics
            recon = reconstruct_leads(leads)

            # Module D: ML Inference (mock)
            model_input = ModelInput(
                signal_tensor=recon.signal_tensor,
                lead_mask=recon.lead_mask,
                sampling_rate=recon.sampling_rate,
            )
            model_output = self._engine.predict(model_input)

            total_ms = (time.perf_counter() - t0) * 1000.0

            # Dispatch to UI thread
            Clock.schedule_once(
                lambda dt: self._update_ui(
                    frame_bgr, vision_result, leads, recon,
                    model_output, total_ms
                )
            )

        except Exception as e:
            Clock.schedule_once(
                lambda dt: self._show_status(f"Pipeline error: {e}")
            )

    # ------------------------------------------------------------------
    # UI updates (main thread)
    # ------------------------------------------------------------------

    def _update_ui(
        self,
        frame_bgr: np.ndarray,
        vision: VisionResult,
        leads: list,
        recon: ReconstructionResult,
        model_output: ModelOutput,
        total_ms: float,
    ):
        """Update all UI elements from pipeline results."""
        if not self._running:
            return

        # FPS counter
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = now

        # Update camera display
        self._update_display_texture(frame_bgr)

        # Edge case handling
        if not leads:
            self._show_status("No Waveform Detected")
            self._overlay.waveform_points = []
            self._diag_panel.hide()
            return

        avg_snr = np.mean([l.snr_db for l in leads])
        avg_cont = np.mean([l.continuity_pct for l in leads])

        if avg_cont < 50.0:
            self._show_status("Signal Unstable")
        elif vision.calibration_confidence < 0.5:
            self._show_status("⚠ Low Calibration Confidence")
        elif model_output.confidence_score < 0.5:
            self._show_status("Low Confidence — Verify")
        else:
            self._status_label.text = ""

        # Update waveform overlay (use first available lead)
        primary_lead = leads[0]
        self._overlay.update_waveform(
            primary_lead.voltage_mv, primary_lead.time_s
        )

        # Suspicious segments
        self._overlay.is_abnormal = (
            model_output.binary_prediction == "abnormal"
        )
        if model_output.suspicious_segments:
            self._overlay.update_suspicious_segments(
                model_output.suspicious_segments,
                total_samples=len(primary_lead.voltage_mv),
            )
        else:
            self._overlay.update_suspicious_segments([], 0)

        # Diagnostic panel
        self._diag_panel.update_from_model_output(model_output)

        # Debug overlay
        if self.app_config.debug_mode:
            self._debug_overlay.visible = True
            self._debug_overlay.update_stats(
                tensor_shape=recon.signal_tensor.shape,
                lead_mask=[int(m) for m in recon.lead_mask],
                rmse=recon.rmse_reconstruction,
                snr=avg_snr,
                cal_confidence=vision.calibration_confidence,
                latency_ms=total_ms,
                fps=self._current_fps,
            )

    def _update_display_texture(self, frame_bgr: np.ndarray):
        """Convert an OpenCV BGR frame to a Kivy texture and display it."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)  # Kivy expects flipped
        h, w = frame_rgb.shape[:2]

        texture = Texture.create(size=(w, h), colorfmt="rgb")
        texture.blit_buffer(
            frame_rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte"
        )
        self._display.texture = texture

    def _show_status(self, text: str):
        """Display an edge-case status message."""
        self._status_label.text = text
