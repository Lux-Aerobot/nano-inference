#!/bin/bash
#export OPENBLAS_CORETYPE=ARMV8
rm logs.log
rm stats.log
rm -rf results
python3 stats.py & python3 ml_onnx.py 'test_images'
