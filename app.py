"""
NeuroScope AI — Kivy Application Entry Point

Usage:
    python app.py                          # Camera mode
    python app.py --test-image ecg.jpg     # Test image mode
    python app.py --debug                  # Enable debug overlay
    python app.py --theme green_on_black   # Override theme

LANDSCAPE: Window is set to 800×400 (landscape) by default.
"""

import sys
import os
import argparse
import logging

os.environ["KIVY_NO_ARGS"] = "1"
os.environ.setdefault("KIVY_LOG_LEVEL", "info")

from kivy.config import Config
# Landscape dimensions
Config.set("graphics", "width",     "800")
Config.set("graphics", "height",    "400")
Config.set("graphics", "resizable", "1")
Config.set("kivy",     "window_icon", "")

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.core.window import Window

from config import AppConfig
from ui.screens.settings_screen import SettingsScreen
from ui.screens.main_screen     import MainScreen
from ui.styles import BG_DARK

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("NeuroScope")


class NeuroScopeApp(App):
    """
    NeuroScope AI — Real-time ECG digitization and assistive analysis.

    Flow:
        1. SettingsScreen → select ECG theme + toggles  (landscape)
        2. MainScreen     → camera/test-image analysis  (landscape)
    """

    title = "NeuroScope AI"

    def __init__(self, test_image: str = None, debug: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._test_image = test_image
        self._debug_override = debug
        self.app_config = AppConfig.load()
        if debug:
            self.app_config.debug_mode = True

    def build(self):
        Window.clearcolor = BG_DARK

        sm = ScreenManager(transition=FadeTransition(duration=0.3))

        settings = SettingsScreen(app_config=self.app_config)
        sm.add_widget(settings)

        main = MainScreen(
            app_config=self.app_config,
            test_image=self._test_image,
        )
        sm.add_widget(main)

        sm.current = "settings"
        return sm

    def on_stop(self):
        self.app_config.save()
        log.info("Config saved. Goodbye!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NeuroScope AI — ECG Digitization App")
    parser.add_argument("--test-image", type=str, default=None,
                        help="Path to a test ECG image (bypasses camera)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug overlay on startup")
    parser.add_argument(
        "--theme",
        type=str,
        # blue_on_white removed from user-selectable themes
        choices=["green_on_black", "white_on_black", "black_on_white", "yellow_on_dark"],
        default=None,
        help="Override ECG theme",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.theme:
        config = AppConfig.load()
        config.theme = args.theme
        config.save()

    app = NeuroScopeApp(test_image=args.test_image, debug=args.debug)
    app.run()


if __name__ == "__main__":
    main()