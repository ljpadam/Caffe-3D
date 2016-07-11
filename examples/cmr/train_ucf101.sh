#!/usr/bin/env sh

./build/tools/caffe \
  train \
  --solver=examples/cmr/solver.prototxt \
  2>&1 | tee examples/cmr/c3d_ucf101_train.log
