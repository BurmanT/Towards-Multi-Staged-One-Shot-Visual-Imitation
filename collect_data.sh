#!/bin/bash
SUITE=/home/tburma01/mosaic/presentation

TASK_name=pick_place
N_VARS=1 # number of variations for this task
NUM=3
for ROBOT in panda sawyer
do 
python tasks/collect_data/collect_any_task.py ${SUITE}/${TASK_name}/${ROBOT}_${TASK_name} \
  -tsk ${TASK_name} -ro ${ROBOT} --n_tasks ${N_VARS}  \
   --N ${NUM} --per_task_group 100 --num_workers 20 --collect_cam  --heights 100 --widths 180
done 
