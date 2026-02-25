"""
NeuroOverlay — Waveform overlay widget.

Draws the extracted ECG waveform as a thin line on the Kivy canvas,
with suspicious segment highlighting (pulsing red glow rectangles).
"""

from kivy.uix.widget import Widget
from kivy.graphics import (
    Color, Line, Rectangle, RoundedRectangle,
    PushMatrix, PopMatrix, Rotate, Scale,
)
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.properties import (
    ListProperty, BooleanProperty, NumericProperty, ObjectProperty,
)
import numpy as np

from ui.styles import (
    WAVEFORM_COLOR, WAVEFORM_NORMAL, ACCENT_RED_GLOW,
    PULSE_DURATION, PULSE_ALPHA_MIN, PULSE_ALPHA_MAX,
)


class NeuroOverlay(Widget):
    """
    Transparent overlay that draws:
      1. ECG waveform trace (thin colored line)
      2. Suspicious segment rectangles with pulsing animation
      3. Gradient red transparency for abnormal regions
    """

    # Waveform data: flat list of (x, y, x, y, ...) for Kivy Line
    waveform_points = ListProperty([])

    # Suspicious segments: list of dicts {"start_x": float, "end_x": float,
    #                                     "y_top": float, "y_bottom": float}
    suspicious_segments = ListProperty([])

    # Whether the current prediction is abnormal
    is_abnormal = BooleanProperty(False)

    # Pulsing opacity for suspicious regions
    _pulse_alpha = NumericProperty(PULSE_ALPHA_MIN)

    # Status text (e.g. "Signal Unstable")
    status_text = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pulse_anim = None
        self.bind(
            waveform_points=self._redraw,
            suspicious_segments=self._redraw,
            is_abnormal=self._redraw,
            pos=self._redraw,
            size=self._redraw,
            _pulse_alpha=self._redraw_segments_only,
        )

    def start_pulse(self):
        """Start the pulsing animation for suspicious segments."""
        if self._pulse_anim:
            self._pulse_anim.cancel(self)

        anim_up = Animation(
            _pulse_alpha=PULSE_ALPHA_MAX,
            duration=PULSE_DURATION / 2,
            t="in_out_sine",
        )
        anim_down = Animation(
            _pulse_alpha=PULSE_ALPHA_MIN,
            duration=PULSE_DURATION / 2,
            t="in_out_sine",
        )
        self._pulse_anim = anim_up + anim_down
        self._pulse_anim.repeat = True
        self._pulse_anim.start(self)

    def stop_pulse(self):
        """Stop the pulsing animation."""
        if self._pulse_anim:
            self._pulse_anim.cancel(self)
            self._pulse_anim = None
        self._pulse_alpha = PULSE_ALPHA_MIN

    def update_waveform(self, voltage_mv: np.ndarray, time_s: np.ndarray):
        """
        Convert signal arrays to canvas-space points.

        Maps time_s → horizontal (0 to widget width)
        Maps voltage_mv → vertical (centered at widget height/2,
                                     scaled to fill ~60% of height)
        """
        if len(voltage_mv) == 0 or len(time_s) == 0:
            self.waveform_points = []
            return

        w, h = self.width, self.height
        x0, y0 = self.x, self.y

        # Downsample for rendering performance (max 1000 points)
        max_pts = 1000
        if len(voltage_mv) > max_pts:
            step = len(voltage_mv) // max_pts
            voltage_mv = voltage_mv[::step]
            time_s = time_s[::step]

        # Normalize time → [0, w]
        t_min, t_max = time_s[0], time_s[-1]
        t_range = t_max - t_min if t_max > t_min else 1.0
        x_vals = x0 + (time_s - t_min) / t_range * w

        # Normalize voltage → centered in widget, 60% height
        v_range = max(abs(voltage_mv.max()), abs(voltage_mv.min()), 0.1)
        scale = (h * 0.3) / v_range
        y_center = y0 + h * 0.5
        y_vals = y_center + voltage_mv * scale

        # Flatten to [x0, y0, x1, y1, ...]
        points = []
        for x, y in zip(x_vals, y_vals):
            points.append(float(x))
            points.append(float(y))

        self.waveform_points = points

    def update_suspicious_segments(
        self,
        segments: list,
        total_samples: int,
    ):
        """
        Map model suspicious segments (sample indices) to canvas rectangles.

        Args:
            segments: [{"start": int, "end": int}, ...]
            total_samples: Total number of samples in the signal.
        """
        if not segments or total_samples <= 0:
            self.suspicious_segments = []
            self.stop_pulse()
            return

        w, h = self.width, self.height
        x0, y0 = self.x, self.y
        rects = []
        for seg in segments:
            sx = x0 + (seg["start"] / total_samples) * w
            ex = x0 + (seg["end"] / total_samples) * w
            rects.append({
                "start_x": sx,
                "end_x": ex,
                "y_top": y0 + h * 0.2,
                "y_bottom": y0 + h * 0.8,
            })

        self.suspicious_segments = rects
        self.start_pulse()

    def _redraw(self, *args):
        """Full redraw of waveform and segments."""
        self.canvas.after.clear()
        with self.canvas.after:
            # Draw suspicious segments first (behind waveform)
            if self.is_abnormal and self.suspicious_segments:
                for seg in self.suspicious_segments:
                    Color(0.95, 0.25, 0.25, self._pulse_alpha)
                    Rectangle(
                        pos=(seg["start_x"], seg["y_top"]),
                        size=(
                            seg["end_x"] - seg["start_x"],
                            seg["y_bottom"] - seg["y_top"],
                        ),
                    )

            # Draw waveform line
            if self.waveform_points:
                color = WAVEFORM_COLOR if self.is_abnormal else WAVEFORM_NORMAL
                Color(*color)
                Line(
                    points=self.waveform_points,
                    width=1.5,
                    cap="round",
                    joint="round",
                )

    def _redraw_segments_only(self, *args):
        """Lightweight redraw for pulse animation — only update segment alpha."""
        self._redraw()
