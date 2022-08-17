from Dataset import DatasetAdversarial
import torchvision.transforms as T
import glob
import json
import sys
import math
import torch
import time

def load_config(CONFIG_DATALOADER_PATH):
    CONFIG_DATALOADER = json.load(open(CONFIG_DATALOADER_PATH))
    configs = glob.glob(CONFIG_DATALOADER["EXECUTOR_CONFIGS_PATH"] + "*.json")
    configs.sort()

    CONFIG_EXECUTOR = json.load(open(configs[-1]))

    return [CONFIG_DATALOADER, CONFIG_EXECUTOR]

def getDatasetLoader(CONFIG_DATALOADER_PATH, type_="train", num_workers=0, pin_memory=False):
    CONFIG_DATALOADER, CONFIG_EXECUTOR = load_config(CONFIG_DATALOADER_PATH)
    if(CONFIG_EXECUTOR["BATCH_SIZE"] % CONFIG_DATALOADER["TRAIN_BATCH_SIZE"] != 0):
        raise ValueError('The executor batch size should be divisible by the train batch size....')

    if(type_ == "train"):
        DATA_SET_END = CONFIG_EXECUTOR["DATA_SET_END_INDEX_TRAIN"]
    else:
        DATA_SET_END = CONFIG_EXECUTOR["DATA_SET_END_INDEX_VAL"]
    
    slice_ = int(CONFIG_EXECUTOR["BATCH_SIZE"] / CONFIG_DATALOADER["TRAIN_BATCH_SIZE"])
    len_ = DATA_SET_END / CONFIG_DATALOADER["TRAIN_BATCH_SIZE"]
        
    dataset = None    

    if(type_ == "train"):
        len_ = int(len_)

        dataset = DatasetAdversarial(
            CONFIG_DATALOADER["DATA_QUEUE_PATH_LOADER"],
            len_,
            slice_
        )
    else:
        if(len_ != int(len_)): len_ = int(len_) + 1

        dataset = DatasetAdversarial(
            CONFIG_DATALOADER["DATA_QUEUE_PATH_LOADER"][:-1] + "_val/",
            len_,
            slice_
        )

    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)
