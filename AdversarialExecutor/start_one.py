from Executor import Executor
import sys
import os
import json
import time

def start(CONFIG, CONFIG_NAME):
    def get_freer_gpu(gpu_id=None):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >xtmp')
        memory_used = [int(x.split()[2]) for x in open('xtmp', 'r').readlines()]
        
        print("GPU(s) memory in use:", memory_used)

        free_gpu = []

        if(gpu_id):
            print("Try to use GPU device:" + CONFIG['DEVICE'] + "...")
            if(CONFIG['GPU_MAX_MEMORY_IN_USED'] >= memory_used[int(gpu_id)]):
                print("GPU device:" + CONFIG['DEVICE'] + " is ready to use...")
                free_gpu.append(int(gpu_id))
            else:
                print("GPU device:" + CONFIG['DEVICE'] + " is not free...")
        else:
            for i in range(len(memory_used)):
                if(CONFIG['GPU_MAX_MEMORY_IN_USED'] >= memory_used[i]):
                    print("GPU device:" + "cuda:" + str(i) + " is ready to use...")
                    free_gpu.append(i)

        return free_gpu

    free_gpu = get_freer_gpu(CONFIG['DEVICE'].split(":")[-1])

    while(not len(free_gpu)):
        print("There is no free GPU wait 10(s)...")
        CONFIG = json.load(open(sys.argv[1]))
        
        if(not CONFIG['Allow_TO_RUN']):
            return
        
        time.sleep(10)
        free_gpu = get_freer_gpu(CONFIG['DEVICE'].split(":")[-1])
        
    if(free_gpu[0] == 2):
        free_gpu[0] = 3
    else:
        if(free_gpu[0] == 3):
            free_gpu[0] = 2
        
    DEVICE = 'cuda:' + str(free_gpu[0])
    print("Use GPU device:" + DEVICE)

    Executor(
        config_name=CONFIG_NAME,
        model_cache=CONFIG['MODEL_CACHE'],
        queue_size_train=CONFIG['QUEUE_SIZE_TRAIN'],
        queue_size_val=CONFIG['QUEUE_SIZE_VAL'],
        data_queue=CONFIG['DATA_QUEUE'],
        data_path=CONFIG['DATA_PATH'],
        batch_size=CONFIG['BATCH_SIZE'],
        device=DEVICE,
        number_of_steps=CONFIG['NUMBER_OF_STEPS'],
        data_set_start_index_train=CONFIG['DATA_SET_START_INDEX_TRAIN'],
        data_set_end_index_train=CONFIG['DATA_SET_END_INDEX_TRAIN'],
        data_set_start_index_val=CONFIG['DATA_SET_START_INDEX_VAL'],
        data_set_end_index_val=CONFIG['DATA_SET_END_INDEX_VAL']
    ).start()

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    print("Use config:" + str(sys.argv[1]) + "...")
    CONFIG = json.load(open(sys.argv[1]))
    start(CONFIG, sys.argv[1])
     
