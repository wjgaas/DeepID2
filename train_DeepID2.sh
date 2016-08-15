#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/deepid2/DeepID2_solver.prototxt $@
