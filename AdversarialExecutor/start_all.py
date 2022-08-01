from ConfigGenerator import config_generator
import subprocess
import glob
import json
import sys

def start_all(CONFIG_PATH, script):
    config_generator(CONFIG_PATH)
    configs = glob.glob("../AdversarialExecutor/ExecutorConfigs/*.json")

    for config in configs:
        log_path = json.load(open(config))["LOG_PATH"]
        bashCommand = [script, config, log_path]
        list_files = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    start_all(sys.argv[1], "./start_one.sh")

