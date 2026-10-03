"""
JOINTS X INSPIRE 2026 - Official Submission Packager
Packages [Nama Tim].zip according to Technical Meeting rules (Page 8):
- [Nama Tim].zip
  ├── [Nama Tim].ipynb  (Jury Audit Notebook)
  └── [Nama Tim].pkl    (Trained Model Weights <= 200 MB)
"""

import os
import sys
import io
import shutil
import zipfile
import pickle

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def create_submission_package(team_name="jarvis"):
    print("=" * 80)
    print(f"[*] PACKAGING OFFICIAL SUBMISSION FOR TEAM: {team_name.upper()}")
    print("=" * 80)

    notebook_source = "solution.ipynb"
    weights_source = "weights/leak_free_hurdle_ensemble.pkl"

    if not os.path.exists(notebook_source):
        raise FileNotFoundError(f"Notebook {notebook_source} not found!")
    if not os.path.exists(weights_source):
        raise FileNotFoundError(f"Weights {weights_source} not found!")

    target_nb = f"{team_name}.ipynb"
    target_weights = f"{team_name}.pkl"
    target_zip = f"{team_name}.zip"

    # Copy files with official naming
    shutil.copyfile(notebook_source, target_nb)
    shutil.copyfile(weights_source, target_weights)

    weights_size_mb = os.path.getsize(target_weights) / (1024 * 1024)
    print(f"  • Notebook File: {target_nb} ({os.path.getsize(target_nb)/1024:.1f} KB)")
    print(f"  • Weights File : {target_weights} ({weights_size_mb:.2f} MB)")

    if weights_size_mb > 200.0:
        raise ValueError(f"CRITICAL: Weights size {weights_size_mb:.2f} MB exceeds 200 MB limit!")
    print("  • Rule Compliance: Weights size <= 200 MB PASSED [OK]")

    # Create ZIP archive
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(target_nb, arcname=target_nb)
        zf.write(target_weights, arcname=target_weights)

    zip_size_mb = os.path.getsize(target_zip) / (1024 * 1024)
    print(f"\n[OK] OFFICIAL SUBMISSION ZIP CREATED:")
    print(f"  * Path: {os.path.abspath(target_zip)}")
    print(f"  * Size: {zip_size_mb:.2f} MB")
    print("=" * 80)

    # Clean temporary uncompressed copies to keep folder neat
    if os.path.exists(target_nb): os.remove(target_nb)
    if os.path.exists(target_weights): os.remove(target_weights)

if __name__ == '__main__':
    t_name = sys.argv[1] if len(sys.argv) > 1 else "jarvis"
    create_submission_package(t_name)
