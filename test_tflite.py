import sys
import traceback
from pipeline.model_engine import create_engine

try:
    e = create_engine(backend='tflite', model_path='pipeline/v16_best.tflite')
    print("Ready:", e.is_ready())
except Exception as e:
    traceback.print_exc(file=sys.stdout)
