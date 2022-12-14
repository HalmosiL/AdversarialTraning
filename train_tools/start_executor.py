import json
import sys

sys.path.append('../')
from executor.Executor import Executor

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    print("Use config:" + str(sys.argv[1]) + "...")
    CONFIG = json.load(open(sys.argv[1]))
     
    Executor(
        model_cache=CONFIG['MODEL_CACHE'],
        queue_size_train=CONFIG['QUEUE_SIZE_TRAIN'],
        queue_size_val=CONFIG['QUEUE_SIZE_VAL'],
        data_queue=CONFIG['DATA_QUEUE'],
        data_path=CONFIG['DATA_PATH'],
        batch_size=CONFIG['BATCH_SIZE'],
        train_batch_size=CONFIG['TRAIN_BATCH_SIZE'],
        num_workers=CONFIG['NUMBER_OF_WORKERS_EXECUTOR'],
        device=CONFIG['DEVICE'][0],
        number_of_steps=CONFIG['NUMBER_OF_STEPS'],
        args_dataset=CONFIG['DATASET'],
        step_size=CONFIG['STEPS_SIZE'],
        clip_size=CONFIG['CLIP_SIZE']
    ).start()
