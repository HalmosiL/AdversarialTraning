from Train import train
import subprocess
import json
import sys
import glob
import os

sys.path.insert(0, "../AdversarialLoader/")
from GetDatasetLoader import getDatasetLoader, getNormalDatasetLoader

sys.path.insert(0, "../AdversarialExecutor/")
from start_all import start_all

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    CONFIG_PATH = sys.argv[1]
    CONFIG = json.load(open(CONFIG_PATH, "r+"))
    
    print("Clear model cache...")
    models_in_cache = glob.glob(CONFIG["MODEL_CACHE"] + "*.pt")
    for m in models_in_cache:
        os.remove(m)

    start_all(CONFIG_PATH, "../AdversarialExecutor/start_one.sh")

    train_loader_adversarial = getDatasetLoader(
        CONFIG_PATH,
        type_="train",
        num_workers=CONFIG["NUMBER_OF_WORKERS_DATALOADER"],
        pin_memory=CONFIG["PIN_MEMORY_ALLOWED_DATALOADER"]
    )
    val_loader_adversarial = getDatasetLoader(
        CONFIG_PATH,
        type_="val",
        num_workers=CONFIG["NUMBER_OF_WORKERS_DATALOADER"],
        pin_memory=CONFIG["PIN_MEMORY_ALLOWED_DATALOADER"]
    )

    val_loader = getNormalDatasetLoader(
        CONFIG_PATH,
        type_="val",
        num_workers=CONFIG["NUMBER_OF_WORKERS_DATALOADER"],
        pin_memory=CONFIG["PIN_MEMORY_ALLOWED_DATALOADER"]
    )

    train(CONFIG_PATH, CONFIG, CONFIG["DEVICE_TRAIN"], train_loader_adversarial, val_loader_adversarial, val_loader)
    subprocess.Popen("../AdversarialExecutor/stop_all.sh", stdout=subprocess.PIPE)
