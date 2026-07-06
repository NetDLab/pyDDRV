"""Test-suite config. The OpenMP duplicate-runtime guard matters only when the
optional torch backend gets imported in a conda env whose libomp differs from
torch's bundled one; the core library is torch-free."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
