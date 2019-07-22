#!/usr/bin/env bash
cd benchmark/bench_utils/pyvotkit
python setup.py build_ext --inplace
cd ../../..

cd benchmark/bench_utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../..
