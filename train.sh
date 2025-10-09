#!/bin/bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

nvflare simulator \
  -w $PWD/workspace/Ped_AT \
  -c client3 \
  -gpu 0 \
  jobs/Ped_AT