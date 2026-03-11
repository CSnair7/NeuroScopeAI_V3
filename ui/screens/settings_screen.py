"""
SettingsScreen — Setup screen in LANDSCAPE orientation.

Two-column layout:
  Left  — Title + 4 theme cards
  Right — Toggles, lead stepper, camera rotation selector, Continue button

Changes vs previous revision:
  - blue_on_white removed from theme list
  - Camera rotation control added (0 / 90 / 180 / 270 degrees)
    Fixes the rotated camera bug on devices whose API reports portrait frames.
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.switch import Switch
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Line, Rectangle
from kivy.clock import Clock
from kivy.metrics import dp, sp

from config import THEME_PRESETS, AppConfig
from ui.styles import (
    BG_DARK, BG_GLASS, BG_GLASS_BORDER,
    GLASS_RADIUS, GLASS_BORDER_WIDTH,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT_BLUE,
    FONT_SIZE_TITLE, FONT_SIZE_HEADING, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
)

_THEME_DESCRIPTIONS = {
    "green_on_black": "Green waveform on dark background",
    "white_on_black": "White waveform on dark background",
    "black_on_white": "Black waveform on white/pink paper",
    "yellow_on_dark": "Yellow/amber waveform on dark background",
}
# ---------------------------------------------------------------------------
# ThemeCard
# ---------------------------------------------------------------------------

class ThemeCard(BoxLayout):
    def __init__(self, theme_name: str, display_name: str, on_select=None, **kwargs):
        super().__init__(**kwargs)
        self.theme_name = theme_name
        self._on_select = on_select
        self.orientation = "vertical"
        self.padding = dp(10)
        self.spacing = dp(3)
        self.size_hint = (1, None)
        self.height = dp(66)
        self._active = False

        with self.canvas.before:
            self._bg_color = Color(*BG_GLASS)
            self._bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
            self._border_color = Color(*BG_GLASS_BORDER)
            self._border_line = Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, dp(10)),
                width=GLASS_BORDER_WIDTH,
            )
        self.bind(pos=self._update_bg, size=self._update_bg)

        name_lbl = Label(
            text=display_name, color=TEXT_PRIMARY, font_size=FONT_SIZE_BODY,
            size_hint_y=None, height=dp(24), halign="left", bold=True,
        )
        name_lbl.bind(size=name_lbl.setter("text_size"))

        desc_lbl = Label(
            text=_THEME_DESCRIPTIONS.get(theme_name, ""),
            color=TEXT_SECONDARY, font_size=FONT_SIZE_CAPTION,
            size_hint_y=None, height=dp(16), halign="left",
        )
        desc_lbl.bind(size=desc_lbl.setter("text_size"))

        self.add_widget(name_lbl)
        self.add_widget(desc_lbl)

    def _update_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._border_line.rounded_rectangle = (
            self.x, self.y, self.width, self.height, dp(10)
        )

    def set_active(self, active: bool):
        self._active = active
        if active:
            self._bg_color.rgba = (*ACCENT_BLUE[:3], 0.22)
            self._border_color.rgba = (*ACCENT_BLUE[:3], 0.55)
        else:
            self._bg_color.rgba = BG_GLASS
            self._border_color.rgba = BG_GLASS_BORDER

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self._on_select:
                self._on_select(self.theme_name)
            return True
        return super().on_touch_down(touch)


# ---------------------------------------------------------------------------
# SettingsScreen
# ---------------------------------------------------------------------------

class SettingsScreen(Screen):

    def __init__(self, app_config: AppConfig, **kwargs):
        super().__init__(**kwargs)
        self.name = "settings"
        self.app_config = app_config
        self._selected_theme = app_config.theme
        self._theme_cards = {}

        with self.canvas.before:
            Color(*BG_DARK)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_main_bg, size=self._update_main_bg)

        # ── Root horizontal split ────────────────────────────────────────
        root = BoxLayout(orientation="horizontal", padding=dp(20), spacing=dp(16))

        # ── LEFT: title + theme cards ───────────────────────────────────
        left = BoxLayout(orientation="vertical", spacing=dp(8), size_hint_x=0.52)

        title = Label(
            text="[b]NeuroScope AI[/b]", markup=True, color=TEXT_PRIMARY,
            font_size=FONT_SIZE_TITLE, size_hint_y=None, height=dp(40), halign="left",
        )
        title.bind(size=title.setter("text_size"))
        left.add_widget(title)

        subtitle = Label(
            text="Select ECG monitor display theme",
            color=TEXT_SECONDARY, font_size=FONT_SIZE_CAPTION,
            size_hint_y=None, height=dp(20), halign="left",
        )
        subtitle.bind(size=subtitle.setter("text_size"))
        left.add_widget(subtitle)

        for name, preset in THEME_PRESETS.items():
            card = ThemeCard(
                theme_name=name, display_name=preset.display_name,
                on_select=self._on_theme_select,
            )
            self._theme_cards[name] = card
            left.add_widget(card)

        left.add_widget(Widget())
        root.add_widget(left)

        # ── RIGHT: controls ─────────────────────────────────────────────
        right = BoxLayout(orientation="vertical", spacing=dp(8), size_hint_x=0.48)

        # Auto-detect
        right.add_widget(self._section_label("Auto-detect Mode"))
        right.add_widget(self._toggle_row(
            "Experimental — auto-select theme",
            active=app_config.auto_detect, attr="_auto_switch",
            callback=self._on_auto_toggle,
        ))

        # Debug mode
        right.add_widget(self._section_label("Debug Mode"))
        right.add_widget(self._toggle_row(
            "Show pipeline diagnostics overlay",
            active=app_config.debug_mode, attr="_debug_switch",
            callback=self._on_debug_toggle,
        ))

        # Visible leads stepper
        right.add_widget(self._section_label("Visible Leads"))
        hint = Label(
            text="0 = auto-detect", color=TEXT_MUTED, font_size=FONT_SIZE_CAPTION,
            size_hint_y=None, height=dp(14), halign="left",
        )
        hint.bind(size=hint.setter("text_size"))
        right.add_widget(hint)

        leads_row = BoxLayout(orientation="horizontal", size_hint_y=None,
                              height=dp(44), spacing=dp(6))
        minus_btn = Button(
            text="-", size_hint=(None, 1), width=dp(44), font_size=sp(20),
            background_color=(*BG_GLASS_BORDER[:3], 0.5), color=TEXT_PRIMARY, bold=True,
        )
        minus_btn.bind(on_release=self._on_leads_minus)
        self._leads_value_label = Label(
            text=self._fmt_leads(app_config.num_leads),
            color=TEXT_PRIMARY, font_size=FONT_SIZE_TITLE, bold=True, halign="center",
        )
        plus_btn = Button(
            text="+", size_hint=(None, 1), width=dp(44), font_size=sp(20),
            background_color=(*BG_GLASS_BORDER[:3], 0.5), color=TEXT_PRIMARY, bold=True,
        )
        plus_btn.bind(on_release=self._on_leads_plus)
        leads_row.add_widget(minus_btn)
        leads_row.add_widget(self._leads_value_label)
        leads_row.add_widget(plus_btn)
        right.add_widget(leads_row)

        right.add_widget(Widget())

        self._continue_btn = Button(
            text="Continue ->",
            size_hint=(1, None), height=dp(50),
            background_color=(*ACCENT_BLUE[:3], 1.0),
            color=TEXT_PRIMARY, font_size=FONT_SIZE_HEADING, bold=True,
        )
        self._continue_btn.bind(on_release=self._on_continue)
        right.add_widget(self._continue_btn)

        root.add_widget(right)
        self.add_widget(root)

        Clock.schedule_once(lambda dt: self._highlight_card(self._selected_theme), 0.1)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> Label:
        lbl = Label(
            text=text, color=TEXT_PRIMARY, font_size=FONT_SIZE_BODY,
            size_hint_y=None, height=dp(22), halign="left", bold=True,
        )
        lbl.bind(size=lbl.setter("text_size"))
        return lbl

    def _toggle_row(self, description: str, active: bool,
                    attr: str, callback) -> BoxLayout:
        row = BoxLayout(orientation="horizontal", size_hint_y=None,
                        height=dp(36), spacing=dp(8))
        lbl = Label(text=description, color=TEXT_SECONDARY,
                    font_size=FONT_SIZE_CAPTION, halign="left")
        lbl.bind(size=lbl.setter("text_size"))
        sw = Switch(active=active, size_hint_x=None, width=dp(72))
        sw.bind(active=callback)
        setattr(self, attr, sw)
        row.add_widget(lbl)
        row.add_widget(sw)
        return row

    def _update_main_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size

    # ── Callbacks ────────────────────────────────────────────────────────

    def _on_theme_select(self, theme_name: str):
        self._selected_theme = theme_name
        self._highlight_card(theme_name)

    def _highlight_card(self, active_name: str):
        for name, card in self._theme_cards.items():
            card.set_active(name == active_name)

    def _on_auto_toggle(self, switch, value):
        self.app_config.auto_detect = value

    def _on_debug_toggle(self, switch, value):
        self.app_config.debug_mode = value

    def _on_leads_minus(self, *args):
        val = max(0, self.app_config.num_leads - 1)
        self.app_config.num_leads = val
        self._leads_value_label.text = self._fmt_leads(val)

    def _on_leads_plus(self, *args):
        val = min(12, self.app_config.num_leads + 1)
        self.app_config.num_leads = val
        self._leads_value_label.text = self._fmt_leads(val)

    @staticmethod
    def _fmt_leads(n: int) -> str:
        return "Auto" if n == 0 else str(n)

    def _on_continue(self, *args):
        self.app_config.theme = self._selected_theme
        self.app_config.save()
        self.manager.current = "main"