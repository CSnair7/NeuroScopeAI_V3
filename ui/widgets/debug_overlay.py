"""
DebugOverlay — Small corner glass panel for pipeline diagnostics.

Displays: tensor shape, lead mask, RMSE, SNR, calibration confidence, latency.
Toggled via settings or triple-tap.
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle, Line
from kivy.animation import Animation
from kivy.properties import BooleanProperty, DictProperty

from ui.styles import (
    BG_GLASS, BG_GLASS_BORDER, GLASS_RADIUS, GLASS_BORDER_WIDTH,
    TEXT_MUTED, FONT_SIZE_DEBUG, FADE_DURATION,
)


class DebugOverlay(BoxLayout):
    """
    Small semi-transparent corner panel showing pipeline debug info.

    Set `debug_data` dict to update all fields at once.
    Toggle `visible` to show/hide with animation.
    """

    visible = BooleanProperty(False)
    debug_data = DictProperty({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 8
        self.spacing = 2
        self.size_hint = (0.45, None)
        self.height = 160
        self.pos_hint = {"right": 0.98, "top": 0.98}
        self.opacity = 0

        # Glass background
        with self.canvas.before:
            self._bg_color = Color(*BG_GLASS)
            self._bg_rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=GLASS_RADIUS
            )
            self._border_color = Color(*BG_GLASS_BORDER)
            self._border_line = Line(
                rounded_rectangle=(
                    self.x, self.y, self.width, self.height,
                    GLASS_RADIUS[0],
                ),
                width=GLASS_BORDER_WIDTH,
            )

        self.bind(pos=self._update_bg, size=self._update_bg)

        # Title
        self._title = Label(
            text="[b]DEBUG[/b]",
            markup=True,
            color=TEXT_MUTED,
            font_size=FONT_SIZE_DEBUG,
            size_hint_y=None,
            height=14,
            halign="left",
        )
        self._title.bind(size=self._title.setter("text_size"))
        self.add_widget(self._title)

        # Data labels
        self._labels = {}
        fields = [
            "tensor_shape",
            "lead_mask",
            "rmse",
            "snr",
            "cal_confidence",
            "latency",
            "fps",
        ]
        for field in fields:
            lbl = Label(
                text=f"{field}: --",
                color=TEXT_MUTED,
                font_size=FONT_SIZE_DEBUG,
                size_hint_y=None,
                height=14,
                halign="left",
            )
            lbl.bind(size=lbl.setter("text_size"))
            self._labels[field] = lbl
            self.add_widget(lbl)

        self.bind(visible=self._on_visibility)
        self.bind(debug_data=self._on_data_update)

    def _update_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._border_line.rounded_rectangle = (
            self.x, self.y, self.width, self.height,
            GLASS_RADIUS[0],
        )

    def _on_visibility(self, *args):
        target = 1.0 if self.visible else 0.0
        Animation(opacity=target, duration=FADE_DURATION, t="out_cubic").start(self)

    def _on_data_update(self, *args):
        data = self.debug_data
        self._labels["tensor_shape"].text = (
            f"Tensor: {data.get('tensor_shape', '--')}"
        )
        self._labels["lead_mask"].text = (
            f"Leads: {data.get('lead_mask', '--')}"
        )
        self._labels["rmse"].text = (
            f"RMSE: {data.get('rmse', '--')}"
        )
        self._labels["snr"].text = (
            f"SNR: {data.get('snr', '--')}"
        )
        self._labels["cal_confidence"].text = (
            f"Cal: {data.get('cal_confidence', '--')}"
        )
        self._labels["latency"].text = (
            f"Latency: {data.get('latency', '--')}"
        )
        self._labels["fps"].text = (
            f"FPS: {data.get('fps', '--')}"
        )

    def update_stats(
        self,
        tensor_shape=None,
        lead_mask=None,
        rmse=None,
        snr=None,
        cal_confidence=None,
        latency_ms=None,
        fps=None,
    ):
        """Convenience setter for all debug fields."""
        self.debug_data = {
            "tensor_shape": str(tensor_shape) if tensor_shape is not None else "--",
            "lead_mask": str(lead_mask) if lead_mask is not None else "--",
            "rmse": f"{rmse:.4f}" if rmse is not None else "--",
            "snr": f"{snr:.1f} dB" if snr is not None else "--",
            "cal_confidence": f"{cal_confidence:.2f}" if cal_confidence is not None else "--",
            "latency": f"{latency_ms:.0f} ms" if latency_ms is not None else "--",
            "fps": f"{fps:.1f}" if fps is not None else "--",
        }
