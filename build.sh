#usr/bin/bash

make_libs () {
  uv run setup.py build_ext --inplace;
}

make_libs
