"""
SettingsScreen — First-time setup screen.

Allows the user to select the ECG monitor background theme
and configure auto-detection mode.
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.switch import Switch
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Line, Rectangle
from kivy.animation import Animation
from kivy.clock import Clock

from config import THEME_PRESETS, AppConfig
from ui.styles import (
    BG_DARK, BG_GLASS, BG_GLASS_BORDER, BG_GLASS_HOVER,
    GLASS_RADIUS, GLASS_BORDER_WIDTH,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT_GREEN, ACCENT_BLUE,
    FONT_SIZE_TITLE, FONT_SIZE_HEADING, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
    FADE_DURATION,
)


class ThemeCard(BoxLayout):
    """
    Selectable card representing one ECG background theme.
    Glassmorphism-styled with active state highlighting.
    """

    def __init__(self, theme_name: str, display_name: str,
                 on_select=None, **kwargs):
        super().__init__(**kwargs)
        self.theme_name = theme_name
        self._on_select = on_select
        self.orientation = "vertical"
        self.padding = 12
        self.spacing = 4
        self.size_hint = (1, None)
        self.height = 80
        self._active = False

        # Glass background
        with self.canvas.before:
            self._bg_color = Color(*BG_GLASS)
            self._bg_rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[12]
            )
            self._border_color = Color(*BG_GLASS_BORDER)
            self._border_line = Line(
                rounded_rectangle=(
                    self.x, self.y, self.width, self.height, 12
                ),
                width=GLASS_BORDER_WIDTH,
            )

        self.bind(pos=self._update_bg, size=self._update_bg)

        # Theme name label
        self._name_label = Label(
            text=display_name,
            color=TEXT_PRIMARY,
            font_size=FONT_SIZE_HEADING,
            size_hint_y=None,
            height=30,
            halign="left",
            bold=True,
        )
        self._name_label.bind(size=self._name_label.setter("text_size"))

        # Description label
        desc_map = {
            "green_on_black": "Green waveform on dark background",
            "white_on_black": "White waveform on dark background",
            "black_on_white": "Black waveform on white/pink paper",
            "yellow_on_dark": "Yellow/amber waveform on dark background",
        }
        self._desc_label = Label(
            text=desc_map.get(theme_name, ""),
            color=TEXT_SECONDARY,
            font_size=FONT_SIZE_CAPTION,
            size_hint_y=None,
            height=20,
            halign="left",
        )
        self._desc_label.bind(size=self._desc_label.setter("text_size"))

        self.add_widget(self._name_label)
        self.add_widget(self._desc_label)

    def _update_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._border_line.rounded_rectangle = (
            self.x, self.y, self.width, self.height, 12
        )

    def set_active(self, active: bool):
        self._active = active
        if active:
            self._bg_color.rgba = (*ACCENT_BLUE[:3], 0.20)
            self._border_color.rgba = (*ACCENT_BLUE[:3], 0.5)
        else:
            self._bg_color.rgba = BG_GLASS
            self._border_color.rgba = BG_GLASS_BORDER

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self._on_select:
                self._on_select(self.theme_name)
            return True
        return super().on_touch_down(touch)


class SettingsScreen(Screen):
    """
    First screen: ECG background theme selector.
    """

    def __init__(self, app_config: AppConfig, **kwargs):
        super().__init__(**kwargs)
        self.name = "settings"
        self.app_config = app_config
        self._selected_theme = app_config.theme
        self._theme_cards = {}

        # Dark background
        with self.canvas.before:
            Color(*BG_DARK)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_main_bg, size=self._update_main_bg)

        # Root layout
        root = BoxLayout(
            orientation="vertical",
            padding=24,
            spacing=16,
        )

        # Title
        title = Label(
            text="[b]NeuroScope AI[/b]",
            markup=True,
            color=TEXT_PRIMARY,
            font_size=FONT_SIZE_TITLE,
            size_hint_y=None,
            height=40,
        )
        root.add_widget(title)

        # Subtitle
        subtitle = Label(
            text="Select your ECG monitor's display theme",
            color=TEXT_SECONDARY,
            font_size=FONT_SIZE_BODY,
            size_hint_y=None,
            height=30,
        )
        root.add_widget(subtitle)

        # Theme cards
        cards_layout = BoxLayout(
            orientation="vertical",
            spacing=10,
            size_hint_y=None,
        )
        cards_layout.bind(
            minimum_height=cards_layout.setter("height")
        )

        for name, preset in THEME_PRESETS.items():
            card = ThemeCard(
                theme_name=name,
                display_name=preset.display_name,
                on_select=self._on_theme_select,
            )
            self._theme_cards[name] = card
            cards_layout.add_widget(card)

        root.add_widget(cards_layout)

        # Auto-detect toggle
        auto_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=50,
            padding=(0, 8),
        )
        auto_label = Label(
            text="Auto-detect mode (Experimental)",
            color=TEXT_SECONDARY,
            font_size=FONT_SIZE_BODY,
            halign="left",
        )
        auto_label.bind(size=auto_label.setter("text_size"))
        self._auto_switch = Switch(active=app_config.auto_detect)
        self._auto_switch.bind(active=self._on_auto_toggle)
        auto_row.add_widget(auto_label)
        auto_row.add_widget(self._auto_switch)
        root.add_widget(auto_row)

        # --- Number of leads selector ---
        leads_section_label = Label(
            text="Number of Visible Leads",
            color=TEXT_PRIMARY,
            font_size=FONT_SIZE_BODY,
            size_hint_y=None,
            height=28,
            halign="left",
            bold=True,
        )
        leads_section_label.bind(size=leads_section_label.setter("text_size"))
        root.add_widget(leads_section_label)

        leads_hint = Label(
            text="Set to 0 for auto-detect",
            color=TEXT_MUTED,
            font_size=FONT_SIZE_CAPTION,
            size_hint_y=None,
            height=18,
            halign="left",
        )
        leads_hint.bind(size=leads_hint.setter("text_size"))
        root.add_widget(leads_hint)

        leads_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=50,
            spacing=8,
        )

        # Minus button
        minus_btn = Button(
            text="−",
            size_hint=(None, 1),
            width=50,
            font_size="22sp",
            background_color=(*BG_GLASS_BORDER[:3], 0.5),
            color=TEXT_PRIMARY,
            bold=True,
        )
        minus_btn.bind(on_release=self._on_leads_minus)

        # Current value display
        self._leads_value_label = Label(
            text=self._format_leads_value(app_config.num_leads),
            color=TEXT_PRIMARY,
            font_size=FONT_SIZE_TITLE,
            bold=True,
            halign="center",
        )

        # Plus button
        plus_btn = Button(
            text="+",
            size_hint=(None, 1),
            width=50,
            font_size="22sp",
            background_color=(*BG_GLASS_BORDER[:3], 0.5),
            color=TEXT_PRIMARY,
            bold=True,
        )
        plus_btn.bind(on_release=self._on_leads_plus)

        leads_row.add_widget(minus_btn)
        leads_row.add_widget(self._leads_value_label)
        leads_row.add_widget(plus_btn)
        root.add_widget(leads_row)

        # Debug mode toggle
        debug_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=50,
            padding=(0, 8),
        )
        debug_label = Label(
            text="Debug Mode",
            color=TEXT_MUTED,
            font_size=FONT_SIZE_BODY,
            halign="left",
        )
        debug_label.bind(size=debug_label.setter("text_size"))
        self._debug_switch = Switch(active=app_config.debug_mode)
        self._debug_switch.bind(active=self._on_debug_toggle)
        debug_row.add_widget(debug_label)
        debug_row.add_widget(self._debug_switch)
        root.add_widget(debug_row)

        # Spacer
        root.add_widget(Widget())

        # Continue button
        self._continue_btn = Button(
            text="Continue",
            size_hint=(1, None),
            height=50,
            background_color=(*ACCENT_BLUE[:3], 1.0),
            color=TEXT_PRIMARY,
            font_size=FONT_SIZE_HEADING,
            bold=True,
        )
        self._continue_btn.bind(on_release=self._on_continue)
        root.add_widget(self._continue_btn)

        self.add_widget(root)

        # Highlight initial selection
        Clock.schedule_once(lambda dt: self._highlight_card(self._selected_theme), 0.1)

    def _update_main_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size

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
        self._leads_value_label.text = self._format_leads_value(val)

    def _on_leads_plus(self, *args):
        val = min(12, self.app_config.num_leads + 1)
        self.app_config.num_leads = val
        self._leads_value_label.text = self._format_leads_value(val)

    @staticmethod
    def _format_leads_value(n: int) -> str:
        return "Auto" if n == 0 else str(n)

    def _on_continue(self, *args):
        self.app_config.theme = self._selected_theme
        self.app_config.save()
        self.manager.current = "main"
