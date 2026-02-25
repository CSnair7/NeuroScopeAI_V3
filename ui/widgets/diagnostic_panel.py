"""
DiagnosticPanel — Glassmorphism floating panel for predictions.

Displays:
  - Abnormal: Top 2 predictions + confidence + entropy bar
  - Normal: "Normal Sinus Rhythm — XX%" + confidence
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Line, Rectangle
from kivy.animation import Animation
from kivy.properties import (
    StringProperty, NumericProperty, BooleanProperty,
    DictProperty, ListProperty,
)
from kivy.clock import Clock

from ui.styles import (
    BG_GLASS, BG_GLASS_BORDER, GLASS_RADIUS, GLASS_BORDER_WIDTH,
    ACCENT_RED, ACCENT_GREEN, ACCENT_AMBER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_SIZE_HEADING, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
    FADE_DURATION,
)


class EntropyBar(Widget):
    """Visual entropy/uncertainty indicator — horizontal bar."""
    value = NumericProperty(0.5)  # 0 = certain, 1 = uncertain

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = 8
        self.bind(pos=self._draw, size=self._draw, value=self._draw)
        Clock.schedule_once(self._draw, 0)

    def _draw(self, *args):
        self.canvas.clear()
        with self.canvas:
            # Background track
            Color(1, 1, 1, 0.15)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[4])

            # Filled portion
            fill_width = self.width * min(self.value, 1.0)
            if self.value < 0.3:
                Color(*ACCENT_GREEN[:3], 0.8)
            elif self.value < 0.6:
                Color(*ACCENT_AMBER[:3], 0.8)
            else:
                Color(*ACCENT_RED[:3], 0.8)

            RoundedRectangle(
                pos=self.pos,
                size=(fill_width, self.height),
                radius=[4],
            )


class DiagnosticPanel(BoxLayout):
    """
    Semi-transparent floating panel showing ML predictions.

    Properties you set from outside:
      - prediction_text: str — formatted prediction string
      - confidence: float — 0–1
      - is_abnormal: bool
      - entropy: float — 0–1 uncertainty measure
      - status: str — edge case status override ("Signal Unstable", etc.)
    """

    prediction_text = StringProperty("")
    confidence = NumericProperty(0.0)
    is_abnormal = BooleanProperty(False)
    entropy = NumericProperty(0.0)
    status = StringProperty("")  # edge case override

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 16
        self.spacing = 8
        self.size_hint = (0.85, None)
        self.height = 180
        self.pos_hint = {"center_x": 0.5, "y": 0.08}
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

        # Status label (for edge cases)
        self._status_label = Label(
            text="",
            color=ACCENT_AMBER,
            font_size=FONT_SIZE_BODY,
            size_hint_y=None,
            height=24,
            halign="center",
        )
        self._status_label.bind(size=self._status_label.setter("text_size"))

        # Prediction label
        self._pred_label = Label(
            text="",
            color=TEXT_PRIMARY,
            font_size=FONT_SIZE_HEADING,
            markup=True,
            size_hint_y=None,
            height=60,
            halign="left",
            valign="top",
        )
        self._pred_label.bind(size=self._pred_label.setter("text_size"))

        # Confidence label
        self._conf_label = Label(
            text="",
            color=TEXT_SECONDARY,
            font_size=FONT_SIZE_CAPTION,
            size_hint_y=None,
            height=20,
            halign="left",
        )
        self._conf_label.bind(size=self._conf_label.setter("text_size"))

        # Entropy bar
        self._entropy_bar = EntropyBar(size_hint_x=1)

        # Assemble
        self.add_widget(self._status_label)
        self.add_widget(self._pred_label)
        self.add_widget(self._conf_label)
        self.add_widget(self._entropy_bar)

        # Bind property changes
        self.bind(
            prediction_text=self._on_update,
            confidence=self._on_update,
            is_abnormal=self._on_update,
            entropy=self._on_update,
            status=self._on_update,
        )

    def _update_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._border_line.rounded_rectangle = (
            self.x, self.y, self.width, self.height,
            GLASS_RADIUS[0],
        )

    def _on_update(self, *args):
        # Edge-case status override
        if self.status:
            self._status_label.text = self.status
            self._status_label.color = ACCENT_AMBER
        else:
            self._status_label.text = ""

        # Prediction text
        self._pred_label.text = self.prediction_text

        # Confidence
        self._conf_label.text = f"Confidence: {self.confidence:.2f}"

        # Entropy
        self._entropy_bar.value = self.entropy

        # Color accent
        if self.is_abnormal:
            self._pred_label.color = TEXT_PRIMARY
        else:
            self._pred_label.color = ACCENT_GREEN

    def show(self, duration: float = FADE_DURATION):
        """Fade in the panel."""
        Animation(opacity=1.0, duration=duration, t="out_cubic").start(self)

    def hide(self, duration: float = FADE_DURATION):
        """Fade out the panel."""
        Animation(opacity=0.0, duration=duration, t="in_cubic").start(self)

    def update_from_model_output(self, model_output):
        """
        Update the panel from a ModelOutput instance.

        Args:
            model_output: pipeline.model_engine.ModelOutput
        """
        self.is_abnormal = model_output.binary_prediction == "abnormal"
        self.confidence = model_output.confidence_score

        # Compute entropy from class probabilities
        probs = list(model_output.class_probabilities.values())
        probs_arr = [p for p in probs if p > 0]
        if probs_arr:
            ent = -sum(p * (p if p > 0 else 1e-12) for p in probs_arr)
            # Normalize: max entropy for N classes = log(N)
            import math
            max_ent = math.log(max(len(probs_arr), 1))
            self.entropy = min(abs(ent) / max(max_ent, 1e-6), 1.0)
        else:
            self.entropy = 0.0

        # Format prediction text
        sorted_probs = sorted(
            model_output.class_probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        if self.is_abnormal:
            lines = []
            for i, (cls, prob) in enumerate(sorted_probs[:2]):
                lines.append(f"[b]{i+1}. {cls}[/b] — {prob*100:.0f}%")
            self.prediction_text = "\n".join(lines)
        else:
            top_cls, top_prob = sorted_probs[0]
            self.prediction_text = f"[b]{top_cls}[/b] — {top_prob*100:.0f}%"

        self.show()
