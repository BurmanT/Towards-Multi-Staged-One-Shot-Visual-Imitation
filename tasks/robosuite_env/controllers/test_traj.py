import sys
import pickle
from pathlib import Path
sys.path.append("/home/tburma01/mosaic/")

#if str(Path.cwd()) not in sys.path:
#    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pickle as pkl
import imageio

robot = "sawyer"
task = "pick_place"
task_number = "00"

# for each trajectory
for i in range(3):
    objects = []
    if(i<10):
        num = '00' + str(i)
    elif(i < 100):
        num = '0' + str(i)
    else:
        num = str(i)
    #filename = '/home/tburma01/mosaic/multi_stage_dataset/'+task+'/'+robot+'_'+task+"/task_"+task_number+'/'+ 'traj'+num+'.pkl'
    #filename = '/home/tburma01/mosaic/multi_stage_dataset/'+task+'/'+robot+'_'+task+"/practice_pickle/"+'traj0'+num+'.pkl'
    filename = '/home/tburma01/mosaic/presentation/pick_place/sawyer_pick_place/task_00/traj'+num+'.pkl'
    print(filename)
    try:
        with open(filename, "rb") as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
    except:
        print("FILE ABOVE DNE")
        continue
    #print(objects)

    output_dictionary = {}
    #file = open('/data/fangchen/robosuite/new_place_model/results_pick_place/traj'+str(i)+'.pkl','rb')
    #file= open('/home/tburma01/mosaic/mosaic_multitask_dataset/button/panda_button/task_00/traj000.pkl', 'rb')
    len = objects[0]['len']
    env_type = objects[0]['env_type']
    task_id = objects[0]['task_id']
    #print("TASK ID IS ")
    #print(task_id)
    output_dictionary = {'len': len, 'env_type':env_type, 'task_id':task_id}

    #traj = pkl.load(file)['traj']
    traj = objects[0]['traj']
    #out = imageio.get_writer('vis_place/'+str(i)+'.gif')
    for i in range(traj.T):
        output_dictionary[str(i)] = traj.get(i)
        #dictionary = traj.get(i)
        #print(dictionary)
        #print(dictionary.keys())
        #obs = dictionary['obs']
        #print(obs.keys())
        #info = dictionary['info']
        #print(info.keys())
        #output_dictionary[i] = traj.get(i)
        #break
        #obs = traj.get(i)
        #if 'obs' in obs:
        #    img = obs['obs']['image']
        #    out.append_data((img).astype(np.uint8))
    #out.close()

    #print(output_dictionary.keys())

# write the dictionary to the file and pickle it 
#filesave = 'vis_place/'+robot+'/'+task+"/task_"+task_number+'/'+'traj0'+num+'.pkl'
    #filesave = '/home/tburma01/mosaic/multi_stage_dataset/pick_place/panda_pick_place/output_pickle/traj'+num+'.pkl'
    filesave = '/home/tburma01/mosaic/presentation/pick_place/sawyer_pick_place/task_00/latent_traj'+num+'.pkl'
#print(filesave)

    with open(filesave, 'wb') as handle:
        pickle.dump(output_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("done pickling")
