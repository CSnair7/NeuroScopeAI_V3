"""
NeuroOverlay — Multi-lead waveform overlay widget.

Draws extracted ECG waveforms directly on top of the background image,
aligned to the actual lead positions detected in the image. Each lead
trace is drawn at the same vertical position where it appears in the
original ECG image, providing visual confirmation of correct digitization.

Fixes applied (vs original):
  - Bug 1: Unpack 5-tuples from detect_lead_rows correctly
            (y_start, y_end, x_start, x_end, name) not 3-tuple
  - Bug 2: raw_pixel_row is already in full-image absolute coords —
            removed the erroneous second addition of y_start
  - Bug 3: Layout match condition relaxed from == to >= so partial
            lead extraction still uses aligned rendering
  - Bug 4: x_vals now scoped to the cell's x_start/x_end column
  - Bug 5: x_vals now use x_start_clipped/x_end_clipped (the waveform-only
           region after edge-chrome exclusion) instead of the full cell bounds.
           raw_pixel_row has length (x_end_clipped - x_start_clipped), not
           (x_end - x_start), so using the full cell produced a horizontal
           stretch that mapped some data points into the label/chrome zone and
           left others hanging past the right edge — the left/right bleed.
            instead of stretching across the full image width
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
from kivy.metrics import dp

from ui.styles import (
    WAVEFORM_COLOR, WAVEFORM_GREEN, ACCENT_RED_GLOW,
    PULSE_DURATION, PULSE_ALPHA_MIN, PULSE_ALPHA_MAX,
    MONITOR_BG, GRID_COLOR, LEAD_LABEL_COLOR,
    FONT_SIZE_CAPTION, FONT_SIZE_DEBUG,
)




def _to_list(val):
    """Convert numpy array to Python list for safe storage in Kivy ListProperty.
    Kivy compares old vs new ListProperty values with != on every assignment.
    numpy.__ne__ returns an array instead of a bool, causing:
        ValueError: The truth value of an array is ambiguous.
    Plain Python lists compare safely. _redraw converts back with np.asarray().
    """
    if hasattr(val, 'tolist'):
        return val.tolist()
    return val

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

    # Image layout info: (y_start, y_end, x_start, x_end, name) tuples
    # in full-image pixel coords — 5-tuple format from detect_lead_rows.
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

        self._row_by_name = {}  # name → (y_start,y_end,x_start,x_end,name)

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
            lead_rows: list of (y_start, y_end, x_start, x_end, name)
                       5-tuples from vision processor (full-image coords)
            image_shape: (height, width) of the original image
        """
        data = []
        for lead in leads:
            rpr = getattr(lead, 'raw_pixel_row', None)
            data.append({
                "name": lead.name,
                # Convert numpy arrays to Python lists before storing in
                # ListProperty. Kivy's ListProperty compares old vs new value
                # with != on every assignment; numpy.__ne__ returns an array
                # instead of a bool, raising:
                #   ValueError: The truth value of an array is ambiguous.
                # _redraw converts back with np.asarray() for rendering.
                "voltage_mv":      _to_list(lead.voltage_mv),
                "time_s":          _to_list(lead.time_s),
                "raw_pixel_row":   _to_list(rpr),
                "x_start_clipped": getattr(lead, 'x_start_clipped', None),
                "x_end_clipped":   getattr(lead, 'x_end_clipped',   None),
            })

        if image_shape is not None:
            self._image_shape = list(image_shape)
        if lead_rows is not None:
            self._lead_rows = list(lead_rows)
            # Build name→row lookup so _draw_aligned can match by name
            # rather than by index.  This prevents coordinate corruption
            # when extract_leads skips a low-quality cell (continuity<5%),
            # which would shift every subsequent lead's row assignment.
            self._row_by_name = {row[4]: row for row in lead_rows}
        self._lead_data = data

    def update_waveform(self, voltage_mv, time_s):
        """Legacy single-lead API."""
        if len(voltage_mv) == 0:
            self._lead_data = []
            return
        self._lead_data = [{
            "name": "Lead_I",
            "voltage_mv": _to_list(voltage_mv),
            "time_s":     _to_list(time_s),
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

    def _image_col_to_screen_x(self, img_col, img_rect):
        """
        Map an image pixel column to a screen X coordinate.
        """
        rx, _, rw, _ = img_rect
        img_w = self._image_shape[1]
        frac = img_col / img_w
        return rx + rw * frac

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

        # FIX (Bug 3): use >= instead of == so partial lead extraction
        # still gets image-aligned rendering rather than falling back to strips.
        has_layout = (img_rect is not None and len(self._lead_rows) >= n_leads)

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
                # Look up cell bounds by lead name (index-safe).
                # If extract_leads skipped a low-quality cell, lead_data
                # and lead_rows would have different lengths and index i
                # would point to the wrong cell.  Name-based lookup is
                # immune to skipped leads.
                lead_name = lead.get("name", "")
                row_info = self._row_by_name.get(lead_name)
                if row_info is None and i < len(self._lead_rows):
                    row_info = self._lead_rows[i]  # fallback to index
                if row_info is None:
                    continue
                y_start, y_end, x_start, x_end, name = row_info

                # Screen bounds for this lead strip (vertical)
                screen_top = self._image_row_to_screen_y(y_start, img_rect)
                screen_bot = self._image_row_to_screen_y(y_end, img_rect)
                strip_h = abs(screen_top - screen_bot)

                # After the flip, y_start (smaller row) maps to lower screen_y
                # and y_end (larger row) maps to higher screen_y
                strip_bot_screen = min(screen_top, screen_bot)
                strip_top_screen = max(screen_top, screen_bot)

                # Screen bounds for this lead strip (horizontal)
                screen_x_left  = self._image_col_to_screen_x(x_start, img_rect)
                screen_x_right = self._image_col_to_screen_x(x_end,   img_rect)
                strip_w_screen = abs(screen_x_right - screen_x_left)

                if strip_h < 3 or strip_w_screen < 3:
                    continue

                # NOTE: No strip background drawn in aligned mode.
                # The overlay sits directly on the camera image, so filling
                # the cell area with MONITOR_BG would occlude the green ECG
                # trace beneath it, creating the visual illusion that the red
                # overlay trace is displaced from the actual waveform.

                # Waveform trace using raw_pixel_row
                raw_pixel_row = lead.get("raw_pixel_row")
                if raw_pixel_row is not None and len(raw_pixel_row) > 0:
                    # ---- Trim flat tails ----
                    # When the argmax tracker runs past the real ECG waveform
                    # into a uniform-glow region, it locks onto a single row,
                    # producing a perfectly flat tail (constant values). Trim
                    # these from both ends before rendering.
                    rpr = np.asarray(raw_pixel_row, dtype=np.float32)
                    n_full = len(rpr)

                    # Find first index from left where value differs from rpr[0]
                    trim_l = 0
                    if n_full > 10:
                        left_val = rpr[0]
                        for ti in range(1, min(n_full // 4, 200)):
                            if abs(rpr[ti] - left_val) > 1.5:
                                trim_l = ti
                                break
                        else:
                            trim_l = 0  # all same → don't trim

                    # Find first index from right where value differs from rpr[-1]
                    trim_r = n_full
                    if n_full > 10:
                        right_val = rpr[-1]
                        for ti in range(n_full - 2, max(n_full - n_full // 4, n_full - 200), -1):
                            if abs(rpr[ti] - right_val) > 1.5:
                                trim_r = ti + 1
                                break
                        else:
                            trim_r = n_full  # all same → don't trim

                    if trim_r - trim_l < 10:
                        trim_l, trim_r = 0, n_full  # safety: don't over-trim

                    rpr_trimmed = rpr[trim_l:trim_r]

                    # Downsample for rendering
                    n = len(rpr_trimmed)
                    max_pts = 1200
                    if n > max_pts:
                        step = n // max_pts
                        pixel_rows = rpr_trimmed[::step]
                    else:
                        pixel_rows = rpr_trimmed

                    n_pts = len(pixel_rows)

                    # Use the clipped x bounds, narrowed by the flat-tail trim
                    xsc = lead.get("x_start_clipped")
                    xec = lead.get("x_end_clipped")
                    if xsc is not None and xec is not None and xec > xsc:
                        # Adjust col range to account for trimmed samples
                        px_per_col = (xec - xsc) / n_full if n_full > 0 else 1.0
                        col_left  = xsc + trim_l * px_per_col
                        col_right = xsc + (trim_r - 1) * px_per_col
                    else:
                        col_left  = x_start
                        col_right = x_end - 1

                    col_indices = np.linspace(col_left, col_right, n_pts)
                    x_vals = rx + (col_indices / img_w) * rw

                    # FIX (Bug 2): raw_pixel_row is already stored in
                    # full-image absolute row coordinates (signal_processor.py
                    # adds y_start before storing).  The old code added y_start
                    # a second time, shifting every trace down by a full strip.
                    abs_rows = pixel_rows  # already absolute — no +y_start

                    y_vals = np.array([
                        self._image_row_to_screen_y(r, img_rect)
                        for r in abs_rows
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
                        Line(points=points, width=2.0, cap="round", joint="round")

                # Suspicious segments — clipped to cell column
                if self.is_abnormal and self.suspicious_segments:
                    for seg in self.suspicious_segments:
                        seg_x0 = max(seg["start_x"], screen_x_left)
                        seg_x1 = min(seg["end_x"],   screen_x_right)
                        if seg_x1 > seg_x0:
                            Color(0.95, 0.25, 0.25, self._pulse_alpha)
                            Rectangle(
                                pos=(seg_x0, strip_bot_screen),
                                size=(seg_x1 - seg_x0, strip_h),
                            )

        # Lead labels — positioned at top-left of each cell column
        for i, lead in enumerate(self._lead_data):
            lead_name = lead.get("name", "")
            row_info = self._row_by_name.get(lead_name)
            if row_info is None and i < len(self._lead_rows):
                row_info = self._lead_rows[i]
            if row_info is None:
                continue
            y_start, y_end, x_start, x_end, name = row_info

            screen_top = self._image_row_to_screen_y(y_start, img_rect)
            screen_bot = self._image_row_to_screen_y(y_end,   img_rect)
            strip_top_screen = max(screen_top, screen_bot)

            screen_x_left = self._image_col_to_screen_x(x_start, img_rect)

            short = name.replace("Lead_", "") if name.startswith("Lead_") else name
            lbl = Label(
                text=short,
                color=WAVEFORM_GREEN[:3] + (0.9,),
                font_size=FONT_SIZE_CAPTION,
                bold=True,
                size_hint=(None, None),
                size=(dp(40), dp(20)),
                pos=(screen_x_left + dp(4), strip_top_screen - dp(22)),
                halign="left",
                valign="middle",
            )
            lbl.text_size = (dp(40), dp(20))
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
                voltage = np.asarray(lead["voltage_mv"], dtype=np.float32)
                time_s  = np.asarray(lead["time_s"],     dtype=np.float32)
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