import sys
from pathlib import Path
import numpy as np
import pickle as pkl
import imageio

sys.path.append("/home/tburma01/mosaic/")

# for each panda and sawyer
# choose which task
# each trajectory in the task 
robot = "panda"
task = "pick_place"
task_number = "00"
count = 0
for i in range(3):
    #file = open('/home/tburma01/mosaic/mosaic_multitask_dataset/button/sawyer_button/task_00/traj000.pkl', 'rb')
    if(i<10):
        num = '00' + str(i)
    elif (i < 100):
        num = '0' + str(i)
    else:
        num = str(i)
    #filename = '/home/tburma01/mosaic/multi_stage_dataset/'+task+'/'+robot+'_'+task+"/task_"+task_number+'/'+ 'traj'+num+'.pkl'
    filename = '/home/tburma01/mosaic/presentation/pick_place/sawyer_pick_place/task_00/traj'+num+'.pkl'
    try:
        file = open(filename, 'rb')
    except:
        print("this file does not exist")
        continue
    print(filename)
    #filesave = '/home/tburma01/mosaic/multi_stage_dataset/'+task+'/gif_files/'+robot+'/'+task+"/task_"+task_number+'/'+'traj0'+num+'.gif'
    #filesave = '/home/tburma01/mosaic/multi_stage_dataset/'+task+'/'+robot+'_'+task+'/gifs/'+'traj'+num+'.gif'
    filesave = '/home/tburma01/mosaic/presentation/pick_place/sawyer_pick_place/task_00/traj'+num+'.gif'
    print(filesave)
    traj = pkl.load(file)['traj']
    out = imageio.get_writer(filesave)

    for i in range(traj.T):
        obs = traj.get(i)
        if 'obs' in obs:
            img = obs['obs']['image']
            out.append_data((img).astype(np.uint8))
    count += 1
    
    out.close()
    