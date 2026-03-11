import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.system("python scripts/export_tflite.py --checkpoint pipeline/v16_best.pt --num-classes 4 --quantize int8 > debug_output.txt 2>&1")
