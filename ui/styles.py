"""
NeuroScope AI — Glassmorphism style constants and helpers for Kivy.
"""

# ---------------------------------------------------------------------------
# Color palette  (RGBA, 0–1 range)
# ---------------------------------------------------------------------------

# Backgrounds
BG_DARK = (0.05, 0.05, 0.12, 1.0)
BG_GLASS = (1.0, 1.0, 1.0, 0.08)          # glass panel fill
BG_GLASS_HOVER = (1.0, 1.0, 1.0, 0.12)
BG_GLASS_BORDER = (1.0, 1.0, 1.0, 0.20)   # thin white border

# Accent colors
ACCENT_RED = (0.95, 0.25, 0.25, 1.0)       # abnormal highlight
ACCENT_RED_GLOW = (0.95, 0.25, 0.25, 0.35) # suspicious segment glow
ACCENT_GREEN = (0.20, 0.85, 0.45, 1.0)     # normal indicator
ACCENT_AMBER = (1.0, 0.75, 0.0, 1.0)       # low-confidence warning
ACCENT_BLUE = (0.10, 0.45, 0.91, 1.0)      # waveform line (clean)

# Text
TEXT_PRIMARY = (1.0, 1.0, 1.0, 0.95)
TEXT_SECONDARY = (1.0, 1.0, 1.0, 0.60)
TEXT_MUTED = (1.0, 1.0, 1.0, 0.35)
TEXT_DANGER = (0.95, 0.30, 0.30, 0.95)

# Waveform
WAVEFORM_COLOR = (0.95, 0.20, 0.20, 0.90)  # thin red ECG trace
WAVEFORM_NORMAL = (0.10, 0.45, 0.91, 0.90) # clean blue when normal
WAVEFORM_GREEN = (0.20, 0.90, 0.30, 0.95)  # bright green ECG trace

# Monitor display
MONITOR_BG = (0.02, 0.02, 0.06, 0.55)      # semi-transparent background for waveform area
GRID_COLOR = (0.15, 0.15, 0.25, 0.40)       # faint grid lines
LEAD_LABEL_COLOR = (0.60, 0.60, 0.70, 0.85) # lead name labels


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = "Roboto"     # Kivy default; Inter can be swapped via .kv
FONT_SIZE_TITLE = "22sp"
FONT_SIZE_HEADING = "18sp"
FONT_SIZE_BODY = "14sp"
FONT_SIZE_CAPTION = "11sp"
FONT_SIZE_DEBUG = "10sp"


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

GLASS_BORDER_WIDTH = 1.2
GLASS_RADIUS = [16, ]       # corner radius for RoundedRectangle
GLASS_PADDING = 16
PANEL_MARGIN = 12

# Animation
FADE_DURATION = 0.3          # seconds
PULSE_DURATION = 1.5         # seconds for suspicious segment pulse
PULSE_ALPHA_MIN = 0.20
PULSE_ALPHA_MAX = 0.50


# ---------------------------------------------------------------------------
# Helper: KV string for a glass panel
# ---------------------------------------------------------------------------

GLASS_PANEL_KV = """
<GlassPanel@BoxLayout>:
    canvas.before:
        Color:
            rgba: {bg}
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: {radius}
        Color:
            rgba: {border}
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, {radius_val}
            width: {border_w}
""".format(
    bg=str(BG_GLASS),
    border=str(BG_GLASS_BORDER),
    radius=str(GLASS_RADIUS),
    radius_val=GLASS_RADIUS[0],
    border_w=GLASS_BORDER_WIDTH,
)
