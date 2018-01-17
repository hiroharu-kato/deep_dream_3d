#!/usr/bin/env bash

python ./examples/run.py -i ./examples/data/teapot.obj -d ./examples/data -o ./examples/data/teapot.gif
python ./examples/run.py -i ./examples/data/bunny.obj -d ./examples/data -ib [0,0.5,0.5]  -o ./examples/data/bunny.gif
