"""
NeuroOverlay — Multi-lead waveform overlay widget.

Draws extracted ECG waveforms directly on top of the background image,
aligned to the actual lead positions detected in the image. Each lead
trace is drawn at the same vertical position where it appears in the
original ECG image, providing visual confirmation of correct digitization.
"""

from kivy.uix.widget import Widget
from kivy.graphics import (
    Color, Line, Rectangle, RoundedRectangle,
)
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.properties import (
    ListProperty, BooleanProperty, NumericProperty, ObjectProperty,
)
from kivy.uix.label import Label
import numpy as np

from ui.styles import (
    WAVEFORM_COLOR, WAVEFORM_GREEN, ACCENT_RED_GLOW,
    PULSE_DURATION, PULSE_ALPHA_MIN, PULSE_ALPHA_MAX,
    MONITOR_BG, GRID_COLOR, LEAD_LABEL_COLOR,
    FONT_SIZE_CAPTION, FONT_SIZE_DEBUG,
)


class NeuroOverlay(Widget):
    """
    Full-screen overlay that draws extracted ECG waveforms aligned to the
    background ECG image.

    The overlay computes the image-to-screen coordinate transform so that
    each lead's waveform is drawn at the same vertical position where the
    lead appears in the original image.
    """

    # Internal lead data
    _lead_data = ListProperty([])

    # Image layout info: (y_start, y_end, name) tuples in image pixel coords
    _lead_rows = ListProperty([])

    # Original image dimensions (h, w) for coordinate mapping
    _image_shape = ListProperty([0, 0])

    # Suspicious segments
    suspicious_segments = ListProperty([])

    # Whether the current prediction is abnormal
    is_abnormal = BooleanProperty(False)

    # Pulsing opacity for suspicious regions
    _pulse_alpha = NumericProperty(PULSE_ALPHA_MIN)

    # Backwards compat
    waveform_points = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pulse_anim = None
        self._labels = []

        self.bind(
            _lead_data=self._redraw,
            _lead_rows=self._redraw,
            _image_shape=self._redraw,
            suspicious_segments=self._redraw,
            is_abnormal=self._redraw,
            pos=self._redraw,
            size=self._redraw,
            _pulse_alpha=self._redraw,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_leads(self, leads, lead_rows=None, image_shape=None):
        """
        Accept a list of LeadSignal objects and layout info.

        Args:
            leads: list of LeadSignal objects
            lead_rows: list of (y_start, y_end, name) from vision processor
            image_shape: (height, width) of the original image
        """
        data = []
        for lead in leads:
            data.append({
                "name": lead.name,
                "voltage_mv": lead.voltage_mv,
                "time_s": lead.time_s,
                "raw_pixel_row": getattr(lead, 'raw_pixel_row', None),
            })

        if image_shape is not None:
            self._image_shape = list(image_shape)
        if lead_rows is not None:
            self._lead_rows = list(lead_rows)
        self._lead_data = data

    def update_waveform(self, voltage_mv, time_s):
        """Legacy single-lead API."""
        if len(voltage_mv) == 0:
            self._lead_data = []
            return
        self._lead_data = [{
            "name": "Lead_I",
            "voltage_mv": voltage_mv,
            "time_s": time_s,
        }]

    def update_suspicious_segments(self, segments, total_samples):
        if not segments or total_samples <= 0:
            self.suspicious_segments = []
            self.stop_pulse()
            return

        w = self.width
        x0 = self.x
        rects = []
        for seg in segments:
            sx = x0 + (seg["start"] / total_samples) * w
            ex = x0 + (seg["end"] / total_samples) * w
            rects.append({"start_x": sx, "end_x": ex})
        self.suspicious_segments = rects
        self.start_pulse()

    def start_pulse(self):
        if self._pulse_anim:
            self._pulse_anim.cancel(self)
        anim_up = Animation(_pulse_alpha=PULSE_ALPHA_MAX,
                            duration=PULSE_DURATION / 2, t="in_out_sine")
        anim_down = Animation(_pulse_alpha=PULSE_ALPHA_MIN,
                              duration=PULSE_DURATION / 2, t="in_out_sine")
        self._pulse_anim = anim_up + anim_down
        self._pulse_anim.repeat = True
        self._pulse_anim.start(self)

    def stop_pulse(self):
        if self._pulse_anim:
            self._pulse_anim.cancel(self)
            self._pulse_anim = None
        self._pulse_alpha = PULSE_ALPHA_MIN

    # ------------------------------------------------------------------
    # Image-to-screen coordinate mapping
    # ------------------------------------------------------------------

    def _get_image_rect(self):
        """
        Compute the screen rectangle where the image is rendered,
        accounting for aspect-ratio-preserved (keep_ratio) display.

        Returns:
            (img_x, img_y, img_w, img_h) in screen coords, or None.
        """
        if self._image_shape[0] <= 0 or self._image_shape[1] <= 0:
            return None

        img_h_px, img_w_px = self._image_shape[0], self._image_shape[1]
        img_aspect = img_w_px / img_h_px

        widget_w, widget_h = self.width, self.height
        widget_x, widget_y = self.x, self.y
        widget_aspect = widget_w / widget_h if widget_h > 0 else 1.0

        if img_aspect > widget_aspect:
            # Image is wider than widget — letterboxed top/bottom
            render_w = widget_w
            render_h = widget_w / img_aspect
        else:
            # Image is taller than widget — pillarboxed left/right
            render_h = widget_h
            render_w = widget_h * img_aspect

        render_x = widget_x + (widget_w - render_w) / 2
        render_y = widget_y + (widget_h - render_h) / 2

        return (render_x, render_y, render_w, render_h)

    def _image_row_to_screen_y(self, img_row, img_rect):
        """
        Map an image pixel row to a screen Y coordinate.

        cv2.flip(_, 0) just corrects for OpenCV (origin=top-left) vs
        Kivy (origin=bottom-left). The image still displays right-side-up:
          - Image row 0 (top of image) → screen top → high Kivy Y
          - Image row H (bottom of image) → screen bottom → low Kivy Y
        """
        _, ry, _, rh = img_rect
        img_h = self._image_shape[0]
        frac = img_row / img_h  # 0 at top, 1 at bottom
        return ry + rh * (1.0 - frac)  # invert: top → high Y

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _redraw(self, *args):
        self.canvas.after.clear()

        # Remove old labels
        for lbl in self._labels:
            if lbl.parent:
                self.remove_widget(lbl)
        self._labels = []

        w, h = self.width, self.height
        x0, y0 = self.x, self.y

        if w < 10 or h < 10:
            return

        n_leads = len(self._lead_data)
        if n_leads == 0:
            return

        img_rect = self._get_image_rect()
        has_layout = (img_rect is not None and len(self._lead_rows) == n_leads)

        if has_layout:
            self._draw_aligned(img_rect)
        else:
            self._draw_strips()

    def _draw_aligned(self, img_rect):
        """Draw waveforms aligned to the actual image lead positions
        using raw_pixel_row for pixel-perfect trace matching."""
        rx, ry, rw, rh = img_rect
        img_h = self._image_shape[0]
        img_w = self._image_shape[1]

        with self.canvas.after:
            for i, lead in enumerate(self._lead_data):
                y_start, y_end, name = self._lead_rows[i]

                # Screen bounds for this lead strip
                screen_top = self._image_row_to_screen_y(y_start, img_rect)
                screen_bot = self._image_row_to_screen_y(y_end, img_rect)
                strip_h = abs(screen_top - screen_bot)

                # After the flip, y_start (smaller row) maps to lower screen_y
                # and y_end (larger row) maps to higher screen_y
                strip_bot_screen = min(screen_top, screen_bot)
                strip_top_screen = max(screen_top, screen_bot)

                if strip_h < 3:
                    continue

                # Semi-transparent strip background
                Color(*MONITOR_BG)
                Rectangle(
                    pos=(rx, strip_bot_screen),
                    size=(rw, strip_h),
                )

                # Waveform trace using raw_pixel_row
                raw_pixel_row = lead.get("raw_pixel_row")
                if raw_pixel_row is not None and len(raw_pixel_row) > 0:
                    # Downsample for rendering
                    n = len(raw_pixel_row)
                    max_pts = 1200
                    if n > max_pts:
                        step = n // max_pts
                        pixel_rows = raw_pixel_row[::step]
                    else:
                        pixel_rows = raw_pixel_row

                    n_pts = len(pixel_rows)

                    # Map each column to screen X
                    col_indices = np.linspace(0, img_w - 1, n_pts)
                    x_vals = rx + (col_indices / img_w) * rw

                    # Map each centroid row to screen Y
                    # centroid_rows[j] is the row within the strip (0=top, strip_h-1=bottom)
                    # Absolute image row = y_start + centroid_rows[j]
                    abs_rows = y_start + pixel_rows
                    y_vals = np.array([
                        self._image_row_to_screen_y(r, img_rect) for r in abs_rows
                    ])

                    points = []
                    for px, py in zip(x_vals, y_vals):
                        points.append(float(px))
                        points.append(float(py))

                    if len(points) >= 4:
                        if self.is_abnormal:
                            Color(*WAVEFORM_COLOR)
                        else:
                            Color(*WAVEFORM_GREEN)
                        Line(points=points, width=1.5, cap="round", joint="round")

                # Suspicious segments
                if self.is_abnormal and self.suspicious_segments:
                    for seg in self.suspicious_segments:
                        Color(0.95, 0.25, 0.25, self._pulse_alpha)
                        Rectangle(
                            pos=(seg["start_x"], strip_bot_screen),
                            size=(seg["end_x"] - seg["start_x"], strip_h),
                        )

        # Lead labels
        for i, lead in enumerate(self._lead_data):
            y_start, y_end, name = self._lead_rows[i]
            screen_top = self._image_row_to_screen_y(y_start, img_rect)
            screen_bot = self._image_row_to_screen_y(y_end, img_rect)
            strip_top_screen = max(screen_top, screen_bot)

            short = name.replace("Lead_", "") if name.startswith("Lead_") else name
            lbl = Label(
                text=short,
                color=WAVEFORM_GREEN[:3] + (0.9,),
                font_size=FONT_SIZE_CAPTION,
                bold=True,
                size_hint=(None, None),
                size=(40, 20),
                pos=(rx + 4, strip_top_screen - 22),
                halign="left",
                valign="middle",
            )
            lbl.text_size = (40, 20)
            self.add_widget(lbl)
            self._labels.append(lbl)

    def _draw_strips(self):
        """Fallback: draw waveforms as independent strips (no image alignment)."""
        w, h = self.width, self.height
        x0, y0 = self.x, self.y
        n_leads = len(self._lead_data)

        title_h = max(18, h * 0.08)
        strip_area_h = h - title_h - 4
        strip_h = strip_area_h / n_leads
        strip_margin = max(2, strip_h * 0.06)
        label_w = max(36, w * 0.10)
        wave_x0 = x0 + label_w
        wave_w = w - label_w - 6
        strip_area_top = y0 + h - title_h

        with self.canvas.after:
            Color(*MONITOR_BG)
            RoundedRectangle(pos=(x0, y0), size=(w, h), radius=[12])

            Color(0.25, 0.25, 0.35, 0.5)
            Line(rounded_rectangle=(x0, y0, w, h, 12), width=1.0)

            for i, lead in enumerate(self._lead_data):
                strip_top = strip_area_top - i * strip_h
                strip_bot = strip_top - strip_h
                inner_top = strip_top - strip_margin
                inner_bot = strip_bot + strip_margin
                inner_h = inner_top - inner_bot
                center_y = inner_bot + inner_h / 2

                if inner_h < 5:
                    continue

                if i > 0:
                    Color(0.20, 0.20, 0.30, 0.35)
                    Line(points=[x0 + 4, strip_top, x0 + w - 4, strip_top], width=0.7)

                # Grid
                Color(*GRID_COLOR)
                for frac in [0.25, 0.50, 0.75]:
                    gy = inner_bot + inner_h * frac
                    Line(points=[wave_x0, gy, wave_x0 + wave_w, gy], width=0.5)
                for vfrac in [0.2, 0.4, 0.6, 0.8]:
                    vx = wave_x0 + wave_w * vfrac
                    Line(points=[vx, inner_bot, vx, inner_top], width=0.5)

                # Waveform
                voltage = lead["voltage_mv"]
                time_s = lead["time_s"]
                if len(voltage) > 0:
                    max_pts = 800
                    if len(voltage) > max_pts:
                        step = len(voltage) // max_pts
                        voltage = voltage[::step]
                        time_s = time_s[::step]

                    t_min, t_max = time_s[0], time_s[-1]
                    t_range = t_max - t_min if t_max > t_min else 1.0
                    x_vals = wave_x0 + (time_s - t_min) / t_range * wave_w

                    v_range = max(np.abs(voltage).max(), 0.05)
                    scale = (inner_h * 0.40) / v_range
                    y_vals = center_y + voltage * scale

                    points = []
                    for px, py in zip(x_vals, y_vals):
                        points.append(float(px))
                        points.append(float(py))

                    if len(points) >= 4:
                        if self.is_abnormal:
                            Color(*WAVEFORM_COLOR)
                        else:
                            Color(*WAVEFORM_GREEN)
                        Line(points=points, width=1.3, cap="round", joint="round")

        # Labels for strips mode
        for i, lead in enumerate(self._lead_data):
            strip_top = strip_area_top - i * strip_h
            strip_bot = strip_top - strip_h
            inner_top = strip_top - strip_margin
            inner_bot = strip_bot + strip_margin
            inner_h = inner_top - inner_bot

            name = lead["name"]
            short = name.replace("Lead_", "") if name.startswith("Lead_") else name
            lbl = Label(
                text=short, color=LEAD_LABEL_COLOR, font_size=FONT_SIZE_CAPTION,
                bold=True, size_hint=(None, None), size=(label_w - 4, inner_h),
                pos=(x0 + 2, inner_bot), halign="center", valign="middle",
            )
            lbl.text_size = (label_w - 4, inner_h)
            self.add_widget(lbl)
            self._labels.append(lbl)
