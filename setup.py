from setuptools import setup
from Cython.Build import cythonize

lib_files: list[str] = [
    "mega/op/gamma.pyx"
]

setup(
    name="mega number utils", 
    ext_modules=cythonize(lib_files, language_level=3),
    script_args=["build_ext"],
    options={"build_ext": {"inplace": True}},
)
