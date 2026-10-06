import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
from validation_framework import compute_mase

# Load our managerial SOTA OOF predictions
try:
    oof_prob = np.load('weights/managerial_oof_prob_blend.npy')
    oof_z = np.load('weights/managerial_oof_z_blend.npy')
    print("Loaded managerial SOTA arrays successfully!")
except Exception as e:
    print("Error loading managerial weights:", e)
    # Check what weights exist
    import os
    print("Files in weights/:", os.listdir('weights'))
