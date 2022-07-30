from Cityscapes import CitySegmentation
import sys
import json
import os
import glob

def config_generator(CONFIG_PATH):
    try: 
        print("Create config folder...")
        os.mkdir("./ExecutorConfigs")
        print("Folder created successfuly...")
    except OSError as error: 
        print("Folder alredy exist...")

    print("Clear config...")

    for filename in glob.glob('./ExecutorConfigs/*.json'):
        os.unlink(filename)

    print("Clear data queue...")

    CONFIG = json.load(open(CONFIG_PATH))
    NAMES = []

    if(os.path.exists(CONFIG['DATA_QUEUE'])):
        for filename in glob.glob(CONFIG['DATA_QUEUE'] + "*.pt"):
            os.unlink(filename)
    else:
        os.mkdir(CONFIG['DATA_QUEUE'])

    if(os.path.exists(CONFIG['DATA_QUEUE'][:-1] + "_val/")):
        for filename in glob.glob(CONFIG['DATA_QUEUE'] + "*.pt"):
            os.unlink(filename)
    else:
        os.mkdir(CONFIG['DATA_QUEUE'][:-1] + "_val/")


    print("Create Logs folder...")

    if(os.path.exists("./Logs")):
        for filename in glob.glob("./Logs/" + "*.log"):
            os.unlink(filename)
    else:
        os.mkdir("./Logs")

    print("Use config:" + str(sys.argv[1]) + "...")

    train_data_set_len = CitySegmentation(root=CONFIG['DATA_PATH'], split="train").__len__()
    val_data_set_len = CitySegmentation(root=CONFIG['DATA_PATH'], split="val").__len__()

    train_split_step = int(train_data_set_len / CONFIG['NUMBER_OF_EXECUTORS'])
    val_split_step = int(val_data_set_len / CONFIG['NUMBER_OF_EXECUTORS'])

    for i in range(CONFIG['NUMBER_OF_EXECUTORS']):
        name = "./ExecutorConfigs/config_" + str(i + 1) + ".json"

        DATA_SET_START_INDEX_TRAIN = i * train_split_step
        DATA_SET_START_INDEX_VAL = i * val_split_step

        if(i + 1 != CONFIG['NUMBER_OF_EXECUTORS']):
            DATA_SET_END_INDEX_TRAIN = (i + 1) * train_split_step
            DATA_SET_END_INDEX_VAL = (i + 1) * val_split_step
        else:
            DATA_SET_END_INDEX_TRAIN = train_data_set_len
            DATA_SET_END_INDEX_VAL = val_data_set_len

        print("Train Executor(" + str(i + 1) + ")", DATA_SET_START_INDEX_TRAIN, "-" ,DATA_SET_END_INDEX_TRAIN, "range...")
        print("Val Executor(" + str(i + 1) + ") ", DATA_SET_START_INDEX_VAL, "-" ,DATA_SET_END_INDEX_VAL, "range...")

        if(CONFIG['LOG_PATH'] != "None"):
            data = {
                'ID': i + 1,
                'MODEL_CACHE': CONFIG['MODEL_CACHE'],
                'GPU_MAX_MEMORY_IN_USED': CONFIG['GPU_MAX_MEMORY_IN_USED'][i],
                'QUEUE_SIZE_TRAIN': CONFIG['QUEUE_SIZE_TRAIN'],
                'QUEUE_SIZE_VAL': CONFIG['QUEUE_SIZE_VAL'],
                'DATA_QUEUE': CONFIG['DATA_QUEUE'],
                'DATA_PATH': CONFIG['DATA_PATH'],
                'BATCH_SIZE': CONFIG['BATCH_SIZE'],
                'DEVICE': CONFIG['DEVICE'][i],
                'NUMBER_OF_STEPS': CONFIG['NUMBER_OF_STEPS'],
                'DATA_SET_START_INDEX_TRAIN': DATA_SET_START_INDEX_TRAIN,
                'DATA_SET_END_INDEX_TRAIN': DATA_SET_END_INDEX_TRAIN,
                'DATA_SET_START_INDEX_VAL': DATA_SET_START_INDEX_VAL,
                'DATA_SET_END_INDEX_VAL': DATA_SET_END_INDEX_VAL,
                'Allow_TO_RUN': True,
                'LOG_PATH': CONFIG['LOG_PATH'] + "executor_log_" + str(i + 1) + ".log"
            }

            open(CONFIG['LOG_PATH'] + "executor_log_" + str(i + 1) + ".log", "x")
        else:
            data = {
                'ID': i + 1,
                'MODEL_CACHE': CONFIG['MODEL_CACHE'],
                'GPU_MAX_MEMORY_IN_USED': CONFIG['GPU_MAX_MEMORY_IN_USED'][i],
                'QUEUE_SIZE_TRAIN': CONFIG['QUEUE_SIZE_TRAIN'],
                'QUEUE_SIZE_VAL': CONFIG['QUEUE_SIZE_VAL'],
                'DATA_QUEUE': CONFIG['DATA_QUEUE'],
                'DATA_PATH': CONFIG['DATA_PATH'],
                'BATCH_SIZE': CONFIG['BATCH_SIZE'],
                'DEVICE': CONFIG['DEVICE'][i],
                'NUMBER_OF_STEPS': CONFIG['NUMBER_OF_STEPS'],
                'DATA_SET_START_INDEX_TRAIN': DATA_SET_START_INDEX_TRAIN,
                'DATA_SET_END_INDEX_TRAIN': DATA_SET_END_INDEX_TRAIN,
                'DATA_SET_START_INDEX_VAL': DATA_SET_START_INDEX_VAL,
                'DATA_SET_END_INDEX_VAL': DATA_SET_END_INDEX_VAL,
                'Allow_TO_RUN': True,
                'LOG_PATH': '/dev/null'
            }

        with open(name, "w") as fp:
            json.dump(data , fp) 

        NAMES.append(name)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    config_generator(sys.argv[1])
