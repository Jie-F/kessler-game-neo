from setuptools import setup, Extension
from mypyc.build import mypycify
from pathlib import Path
import sys

# Automatically add build_ext --inplace if no arguments are passed
if __name__ == "__main__" and len(sys.argv) == 1:
    sys.argv += ["build_ext", "--inplace"]

INIT_PATH = Path(__file__).parent / "__init__.py"

if INIT_PATH.is_file():
    INIT_PATH.unlink()

setup(
    name="neo_controller",
    ext_modules=mypycify(
        ["neo_controller.py"],
        opt_level="3",          # Highest optimization
        #multi_file=True,        # Can improve performance on some projects
        verbose=False,          # Set to True if you want detailed output
    ),
    zip_safe=False,
)

# Recreate __init__.py after build
if not INIT_PATH.is_file():
    INIT_PATH.touch()
