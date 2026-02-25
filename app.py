"""
NeuroScope AI — Kivy Application Entry Point

Usage:
    python app.py                          # Camera mode
    python app.py --test-image ecg.jpg     # Test image mode
    python app.py --debug                  # Enable debug overlay
"""

import sys
import os
import argparse
import logging

# Disable Kivy's own argument parser so our --test-image / --debug / --theme
# flags don't get intercepted.  Must be set BEFORE importing any kivy module.
os.environ["KIVY_NO_ARGS"] = "1"
os.environ.setdefault("KIVY_LOG_LEVEL", "info")

from kivy.config import Config
Config.set("graphics", "width", "400")
Config.set("graphics", "height", "800")
Config.set("graphics", "resizable", "1")
Config.set("kivy", "window_icon", "")

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.core.window import Window

from config import AppConfig
from ui.screens.settings_screen import SettingsScreen
from ui.screens.main_screen import MainScreen
from ui.styles import BG_DARK

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("NeuroScope")


class NeuroScopeApp(App):
    """
    NeuroScope AI — Real-time ECG digitization and assistive analysis.

    Flow:
        1. SettingsScreen → select ECG theme + toggles
        2. MainScreen → camera/test-image analysis with overlays
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
        # Set window background
        Window.clearcolor = BG_DARK

        # Screen manager with fade transition
        sm = ScreenManager(transition=FadeTransition(duration=0.3))

        # Settings screen (first screen)
        settings = SettingsScreen(app_config=self.app_config)
        sm.add_widget(settings)

        # Main analysis screen
        main = MainScreen(
            app_config=self.app_config,
            test_image=self._test_image,
        )
        sm.add_widget(main)

        sm.current = "settings"
        return sm

    def on_stop(self):
        """Save config on exit."""
        self.app_config.save()
        log.info("Config saved. Goodbye!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuroScope AI — ECG Digitization App"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Path to a test ECG image (bypasses camera)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug overlay on startup",
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["green_on_black", "white_on_black",
                 "black_on_white", "yellow_on_dark"],
        default=None,
        help="Override ECG theme (skips settings screen)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.theme:
        config = AppConfig.load()
        config.theme = args.theme
        config.save()

    app = NeuroScopeApp(
        test_image=args.test_image,
        debug=args.debug,
    )
    app.run()


if __name__ == "__main__":
    main()
