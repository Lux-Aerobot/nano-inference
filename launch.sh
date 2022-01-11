#!/bin/bash
rm logs.log
rm stats.log
rm -rf results
python3 stats.py & python3 ml.py '/home/lux/Documents/nano-inference/images'
