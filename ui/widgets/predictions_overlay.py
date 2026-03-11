"""
PredictionsOverlay — Compact glass panel for real-time model predictions.

Styled identically to DebugOverlay (same glass background, border, font).
Positioned bottom-left so it doesn't overlap the DebugOverlay (top-right).

Displays:
  - Binary result  : NORMAL / ABNORMAL (colour-coded)
  - Confidence     : numeric + mini fill bar
  - Top-2 classes  : class name — XX%
  - Entropy        : uncertainty bar
  - Latency        : inference time in ms
"""

import math

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, Line, RoundedRectangle
from kivy.metrics import dp
from kivy.properties import BooleanProperty, DictProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget

# Import only constants that are confirmed present in ui/styles.py.
# ACCENT_RED is NOT exported — neuro_overlay uses ACCENT_RED_GLOW instead.
# We use ACCENT_RED_GLOW for the abnormal state colour.
from ui.styles import (
    ACCENT_AMBER, ACCENT_GREEN, ACCENT_RED_GLOW,
    BG_GLASS, BG_GLASS_BORDER, FADE_DURATION,
    FONT_SIZE_DEBUG, GLASS_BORDER_WIDTH, GLASS_RADIUS,
    TEXT_MUTED, TEXT_PRIMARY,
)


# ---------------------------------------------------------------------------
# Tiny fill-bar widget
# ---------------------------------------------------------------------------

class _MiniBar(Widget):
    """Horizontal fill bar, height 6dp. value = 0.0–1.0."""
    value = NumericProperty(0.0)
    bar_rgba = (0.3, 0.8, 0.3, 0.85)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(6)
        self.bind(pos=self._draw, size=self._draw, value=self._draw)
        Clock.schedule_once(self._draw, 0)

    def _draw(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1, 0.10)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(3)])
            fill_w = max(0.0, min(1.0, self.value)) * self.width
            Color(*self.bar_rgba)
            RoundedRectangle(pos=self.pos, size=(fill_w, self.height), radius=[dp(3)])


# ---------------------------------------------------------------------------
# PredictionsOverlay
# ---------------------------------------------------------------------------

class PredictionsOverlay(BoxLayout):
    """
    Semi-transparent glass corner panel showing live model output.
    Call update_from_model_output(model_output) each inference cycle.
    Toggle visibility with the `visible` property (animates opacity).
    """

    visible = BooleanProperty(False)
    predictions_data = DictProperty({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(8)
        self.spacing = dp(3)
        self.size_hint = (0.42, None)
        self.height = dp(170)
        self.pos_hint = {"x": 0.02, "y": 0.02}
        self.opacity = 0

        # Glass background — identical to DebugOverlay
        with self.canvas.before:
            self._bg_color = Color(*BG_GLASS)
            self._bg_rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=GLASS_RADIUS
            )
            self._border_color = Color(*BG_GLASS_BORDER)
            self._border_line = Line(
                rounded_rectangle=(
                    self.x, self.y, self.width, self.height, GLASS_RADIUS[0],
                ),
                width=GLASS_BORDER_WIDTH,
            )
        self.bind(pos=self._update_bg, size=self._update_bg)

        # Title
        self._title = self._lbl("[b]PREDICTIONS[/b]", markup=True, h=dp(16))
        self.add_widget(self._title)

        # Binary result (colour-coded, larger)
        self._result_lbl = self._lbl("--", markup=True, h=dp(18), bold=True)
        self.add_widget(self._result_lbl)

        # Confidence text + bar
        self._conf_lbl = self._lbl("Conf: --", h=dp(14))
        self.add_widget(self._conf_lbl)
        self._conf_bar = _MiniBar()
        self.add_widget(self._conf_bar)

        # Top-2 class rows
        self._class_lbls = [self._lbl("--", h=dp(14)) for _ in range(2)]
        for l in self._class_lbls:
            self.add_widget(l)

        # Entropy text + bar
        self._entropy_lbl = self._lbl("Entropy: --", h=dp(14))
        self.add_widget(self._entropy_lbl)
        self._entropy_bar = _MiniBar()
        self._entropy_bar.bar_rgba = (*ACCENT_AMBER[:3], 0.80)
        self.add_widget(self._entropy_bar)

        # Latency
        self._latency_lbl = self._lbl("Latency: --", h=dp(14))
        self.add_widget(self._latency_lbl)

        self.bind(visible=self._on_visibility)
        self.bind(predictions_data=self._on_data_update)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _lbl(text, markup=False, h=None, bold=False):
        l = Label(
            text=text, color=TEXT_MUTED, font_size=FONT_SIZE_DEBUG,
            size_hint_y=None, height=h or dp(15),
            halign="left", markup=markup, bold=bold,
        )
        l.bind(size=l.setter("text_size"))
        return l

    def _update_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._border_line.rounded_rectangle = (
            self.x, self.y, self.width, self.height, GLASS_RADIUS[0],
        )

    def _on_visibility(self, *args):
        Animation(
            opacity=1.0 if self.visible else 0.0,
            duration=FADE_DURATION, t="out_cubic",
        ).start(self)

    # ── data update ───────────────────────────────────────────────────────

    def _on_data_update(self, *args):
        d = self.predictions_data
        if not d:
            return

        is_abnormal = d.get("is_abnormal", False)
        binary      = d.get("binary", "--")
        conf        = float(d.get("confidence", 0.0))
        ent         = float(d.get("entropy", 0.0))
        lat         = d.get("latency_ms")
        classes     = d.get("top_classes", [])

        # Binary result — green = normal, red-glow = abnormal
        if binary == "--":
            colour = TEXT_MUTED[:3]
        elif is_abnormal:
            colour = ACCENT_RED_GLOW[:3]
        else:
            colour = ACCENT_GREEN[:3]
        hex_col = "#{:02X}{:02X}{:02X}".format(
            int(colour[0] * 255), int(colour[1] * 255), int(colour[2] * 255)
        )
        self._result_lbl.text = f"[color={hex_col}][b]{binary}[/b][/color]"

        # Confidence bar colour: green >= 0.7, amber >= 0.5, red-glow < 0.5
        if conf >= 0.7:
            self._conf_bar.bar_rgba = (*ACCENT_GREEN[:3], 0.85)
        elif conf >= 0.5:
            self._conf_bar.bar_rgba = (*ACCENT_AMBER[:3], 0.85)
        else:
            self._conf_bar.bar_rgba = (*ACCENT_RED_GLOW[:3], 0.85)
        self._conf_lbl.text  = f"Conf: {conf * 100:.0f}%"
        self._conf_bar.value = conf

        # Top-2 classes
        for i, lbl in enumerate(self._class_lbls):
            if i < len(classes):
                name, prob = classes[i]
                short = name if len(name) <= 22 else name[:20] + "…"
                lbl.text = f"{i + 1}. {short}  {prob * 100:.0f}%"
            else:
                lbl.text = "--"

        # Entropy
        self._entropy_lbl.text  = f"Entropy: {ent:.3f}"
        self._entropy_bar.value = ent

        # Latency
        self._latency_lbl.text = (
            f"Latency: {lat:.0f} ms" if lat is not None else "Latency: --"
        )

    # ── public API ────────────────────────────────────────────────────────

    def update_from_model_output(self, model_output) -> None:
        """Populate from a pipeline.model_engine.ModelOutput instance."""
        is_abnormal = model_output.binary_prediction == "abnormal"
        conf = float(model_output.confidence_score)

        probs = [p for p in model_output.class_probabilities.values() if p > 0]
        if probs:
            raw_ent = -sum(p * math.log(p + 1e-12) for p in probs)
            max_ent = math.log(max(len(probs), 1))
            entropy = min(abs(raw_ent) / max(max_ent, 1e-6), 1.0)
        else:
            entropy = 0.0

        top_classes = sorted(
            model_output.class_probabilities.items(),
            key=lambda x: x[1], reverse=True,
        )[:2]

        self.predictions_data = {
            "is_abnormal": is_abnormal,
            "binary":      "ABNORMAL" if is_abnormal else "NORMAL",
            "confidence":  conf,
            "entropy":     entropy,
            "top_classes": top_classes,
            "latency_ms":  model_output.inference_latency_ms,
        }