import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.dirname(os.path.abspath(__file__))

import kagglehub

# Download latest version
path = kagglehub.competition_download('datajointsxinspire')

print("Path to competition files:", path)
