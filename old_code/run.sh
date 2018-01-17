#!/usr/bin/env bash

#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case001
#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case002 -emin -30 -emax 30
#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case003 -emin -10 -emax 45
#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case004 -emin -10 -emax 45 -l 1
#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case005 -emin -10 -emax 45 -cd 2.0
#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case006 -ll 1
#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case007
#python train.py -of ../../resource/obj/teapot.obj -g 0 -od ../../data/deep_dream/case008
#
python train.py -of ../../resource/obj/bunny.obj -g 1 -od ../../data/deep_dream/case101 &
python train.py -of ../../resource/obj/bunny.obj -g 1 -od ../../data/deep_dream/case102 -ib [0.5,0,0] &
python train.py -of ../../resource/obj/bunny.obj -g 5 -od ../../data/deep_dream/case103 -ib [0,0.5,0] &
python train.py -of ../../resource/obj/bunny.obj -g 0 -od ../../data/deep_dream/case104 -ib [0,0,0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 0 -od ../../data/deep_dream/case105 -ib [0.5,0.5,0] &
python train.py -of ../../resource/obj/bunny.obj -g 2 -od ../../data/deep_dream/case106 -ib [0,0.5,0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 6 -od ../../data/deep_dream/case107 -ib [0.5,0,0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 0 -od ../../data/deep_dream/case108 -ib [0.5,0.5,0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 0 -od ../../data/deep_dream/case109 -ib [-0.5,0,0] &
python train.py -of ../../resource/obj/bunny.obj -g 0 -od ../../data/deep_dream/case110 -ib [0,-0.5,0] &
python train.py -of ../../resource/obj/bunny.obj -g 2 -od ../../data/deep_dream/case111 -ib [0,0,-0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 5 -od ../../data/deep_dream/case112 -ib [-0.5,-0.5,0] &
python train.py -of ../../resource/obj/bunny.obj -g 6 -od ../../data/deep_dream/case113 -ib [0,-0.5,-0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 1 -od ../../data/deep_dream/case114 -ib [-0.5,0,-0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 1 -od ../../data/deep_dream/case115 -ib [-0.5,-0.5,-0.5] &
python train.py -of ../../resource/obj/bunny.obj -g 0 -od ../../data/deep_dream/case116 -bc 0 &
python train.py -of ../../resource/obj/bunny.obj -g 0 -od ../../data/deep_dream/case117 -bc 0.5 &