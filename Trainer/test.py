from Train import train
import subprocess
import json
import sys

sys.path.insert(0, "../AdversarialLoader/")
from GetDatasetLoader import getDatasetLoader, getNormalDatasetLoader

sys.path.insert(0, "../AdversarialExecutor/")
from start_all import start_all

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    CONFIG_PATH = sys.argv[1]
    CONFIG = json.load(open(CONFIG_PATH, "r+"))

    start_all(CONFIG_PATH, "../AdversarialExecutor/start_one.sh")
