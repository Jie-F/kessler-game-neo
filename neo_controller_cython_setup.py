# setup.py
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from pathlib import Path
import sys

# Automatically add build_ext --inplace if no arguments are passed
if __name__ == "__main__" and len(sys.argv) == 1:
    sys.argv += ["build_ext", "--inplace"]

INIT_PATH = Path(__file__).parent / "__init__.py"

if INIT_PATH.is_file():
    INIT_PATH.unlink()

extensions = [
    Extension("neo_controller", ["neo_controller.pyx"]),
]

setup(
    name="neo_controller",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=True,
        compiler_directives={"boundscheck": True, "wraparound": True, "overflowcheck": True}
    ),
    zip_safe=False,
)

# Recreate __init__.py after build
if not INIT_PATH.is_file():
    INIT_PATH.touch()
