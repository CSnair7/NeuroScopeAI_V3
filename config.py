"""
NeuroScope AI — App-wide configuration and constants.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

_CONFIG_FILE_CACHE = None

def get_config_path():
    global _CONFIG_FILE_CACHE
    if _CONFIG_FILE_CACHE is None:
        if "ANDROID_ARGUMENT" in os.environ:
            try:
                from android.storage import app_storage_path
                _CONFIG_FILE_CACHE = os.path.join(app_storage_path(), "neuroscope_config.json")
            except ImportError:
                _CONFIG_FILE_CACHE = os.path.join(APP_DIR, "neuroscope_config.json")
        else:
            _CONFIG_FILE_CACHE = os.path.join(APP_DIR, "neuroscope_config.json")
    return _CONFIG_FILE_CACHE

# ---------------------------------------------------------------------------
# ECG Background Themes
# ---------------------------------------------------------------------------

@dataclass
class ThemePreset:
    name: str
    display_name: str
    hsv_lower: tuple
    hsv_upper: tuple
    trace_is_bright: bool
    bg_hsv_lower: tuple = (0, 0, 0)
    bg_hsv_upper: tuple = (180, 255, 255)


# NOTE: blue_on_white intentionally removed from THEME_PRESETS (settings UI).
# It is still present in vision_processor.BUILTIN_THEMES for auto-detection.
THEME_PRESETS = {
    "green_on_black": ThemePreset(
        name="green_on_black",
        display_name="Green on Black",
        hsv_lower=(25, 20, 35),
        hsv_upper=(95, 255, 255),
        trace_is_bright=True,
        bg_hsv_lower=(0, 0, 0),
        bg_hsv_upper=(180, 255, 50),
    ),
    "white_on_black": ThemePreset(
        name="white_on_black",
        display_name="White on Black",
        hsv_lower=(0, 0, 180),
        hsv_upper=(180, 40, 255),
        trace_is_bright=True,
        bg_hsv_lower=(0, 0, 0),
        bg_hsv_upper=(180, 255, 60),
    ),
    "black_on_white": ThemePreset(
        name="black_on_white",
        display_name="Black on White",
        hsv_lower=(0, 0, 0),
        hsv_upper=(180, 80, 80),
        trace_is_bright=False,
        bg_hsv_lower=(0, 0, 180),
        bg_hsv_upper=(180, 40, 255),
    ),
    "yellow_on_dark": ThemePreset(
        name="yellow_on_dark",
        display_name="Yellow on Dark",
        hsv_lower=(15, 80, 120),
        hsv_upper=(40, 255, 255),
        trace_is_bright=True,
        bg_hsv_lower=(0, 0, 0),
        bg_hsv_upper=(180, 255, 60),
    ),
}

DEFAULT_THEME = "green_on_black"

# ---------------------------------------------------------------------------
# QC Thresholds
# ---------------------------------------------------------------------------

SNR_THRESHOLD_DB = 15.0
CONTINUITY_THRESHOLD = 95.0
QRS_AMP_RANGE_MV = (0.1, 3.5)
QRS_WIDTH_RANGE_MS = (40, 120)
RR_CV_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

TARGET_FPS = 30
MAX_PIPELINE_LATENCY_MS = 150
MAX_RAM_MB = 200
PIPELINE_SAMPLE_RATE_HZ = 500.0
SIGNAL_TENSOR_LENGTH = 5000
NUM_LEADS = 12

# ---------------------------------------------------------------------------
# Persistent config I/O
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    theme: str = DEFAULT_THEME
    auto_detect: bool = False
    debug_mode: bool = False
    num_leads: int = 3
    sample_rate_hz: float = PIPELINE_SAMPLE_RATE_HZ

    def save(self, path: str = None):
        if path is None:
            path = get_config_path()
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = None) -> "AppConfig":
        if path is None:
            path = get_config_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()


def get_theme_preset(name: str) -> ThemePreset:
    """Return theme preset by name, fallback to default."""
    return THEME_PRESETS.get(name, THEME_PRESETS[DEFAULT_THEME])