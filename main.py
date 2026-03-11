"""
main.py — Required entry point for Buildozer / python-for-android.

On Android the Python interpreter is launched by the Java PythonActivity
with special system arguments that argparse cannot parse.  We detect this
early and launch the Kivy App directly, bypassing the CLI entirely.
"""

import os
import sys


def _is_android() -> bool:
    return "ANDROID_ARGUMENT" in os.environ or "ANDROID_ENTRYPOINT" in os.environ


if __name__ == "__main__" or _is_android():
    if _is_android():
        # Android — launch app directly, skip argparse
        from app import NeuroScopeApp
        NeuroScopeApp().run()
    else:
        # Desktop — full CLI support
        from app import main
        main()
