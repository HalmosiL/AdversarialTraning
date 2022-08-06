from Dataset import DatasetAdversarial
import torchvision.transforms as T
import glob
import json
import sys
import math
import torch
import time

sys.path.insert(0, "../AdversarialExecutor/")

from Cityscapes import CitySegmentation

def load_config(CONFIG_DATALOADER_PATH):
    CONFIG_DATALOADER = json.load(open(CONFIG_DATALOADER_PATH))
    configs = glob.glob(CONFIG_DATALOADER["EXECUTOR_CONFIGS_PATH"] + "*.json")
    configs.sort()

    CONFIG_EXECUTOR = json.load(open(configs[-1]))

    return [CONFIG_DATALOADER, CONFIG_EXECUTOR]

def getNormalDatasetLoader(CONFIG_DATALOADER_PATH, type_="val", num_workers=0, pin_memory=False):
    CONFIG_DATALOADER, _ = load_config(CONFIG_DATALOADER_PATH)

    input_transform = T.Compose([
        T.ToTensor(),
    ])

    dataset = CitySegmentation(
        root=CONFIG_DATALOADER["DATA_PATH"],
        split=type_,
        transform=input_transform,
        start_index=0,
        end_index=None
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG_DATALOADER["TRAIN_BATCH_SIZE"],
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def getDatasetLoader(CONFIG_DATALOADER_PATH, type_="train", num_workers=0, pin_memory=False):
    CONFIG_DATALOADER, CONFIG_EXECUTOR = load_config(CONFIG_DATALOADER_PATH)
    
    if(CONFIG_DATALOADER["TRAIN_BATCH_SIZE"] > CONFIG_EXECUTOR["BATCH_SIZE"]):
        concatenate_number = CONFIG_DATALOADER["TRAIN_BATCH_SIZE"] / CONFIG_EXECUTOR["BATCH_SIZE"]
        slice_ = -1

        if(concatenate_number != int(concatenate_number)):
            raise ValueError('The loader BATCH_SIZE should be divisible by EXECUTOR_BATCH_SIZE...')
    else:
        concatenate_number = 0
        slice_ = int(CONFIG_EXECUTOR["BATCH_SIZE"] / CONFIG_DATALOADER["TRAIN_BATCH_SIZE"])
        
        if(CONFIG_EXECUTOR["BATCH_SIZE"] % CONFIG_DATALOADER["TRAIN_BATCH_SIZE"] != 0):
            raise ValueError('The executor batch size should be divisible by the train batch size....')
        
    dataset = None    
    plus_batch_num = None

    if(type_ == "train"):
        len_ = CONFIG_EXECUTOR["DATA_SET_END_INDEX_TRAIN"] / CONFIG_DATALOADER["BATCH_SIZE"]

        if(len_ != int(len_)):
            len_down = int(len_)
            len_ = math.ceil(len_)
            plus_batch_num = math.ceil((CONFIG_EXECUTOR["DATA_SET_END_INDEX_TRAIN"] - len_down * CONFIG_DATALOADER["TRAIN_BATCH_SIZE"]) / CONFIG_EXECUTOR["BATCH_SIZE"])

        dataset = DatasetAdversarial(
            CONFIG_DATALOADER["DATA_QUEUE_PATH_LOADER"],
            len_,
            int(concatenate_number),
            plus_batch_num,
            slice_
        )
    else:
        len_ = CONFIG_EXECUTOR["DATA_SET_END_INDEX_VAL"] / CONFIG_DATALOADER["BATCH_SIZE"]
        
        if(len_ != int(len_)):
            len_down = int(len_) 
            len_ = math.ceil(len_)
            plus_batch_num = math.ceil((CONFIG_EXECUTOR["DATA_SET_END_INDEX_VAL"] - len_down * CONFIG_DATALOADER["TRAIN_BATCH_SIZE"]) / CONFIG_EXECUTOR["BATCH_SIZE"])
        
        dataset = DatasetAdversarial(
            CONFIG_DATALOADER["DATA_QUEUE_PATH_LOADER"][:-1] + "_val/",
            len_,
            int(concatenate_number),
            plus_batch_num,
            slice_ 
        )

    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    print("Use config:" + str(sys.argv[1]) + "...")
    CONFIGS = load_config(sys.argv[1])
    CONFIG_DATALOADER = CONFIGS[0]

    for i in range(5):
        with open(CONFIG_DATALOADER["EXECUTOR_MAIN_CONFIGS_PATH"], 'r+') as f:
            data_json = json.load(f)

            dataloader_ = getDatasetLoader(sys.argv[1], type_=data_json["MODE"])
            print("Mode:", data_json["MODE"])

            for data in dataloader_:
                a, b = data
                print(a.shape)
                print(b.shape)
                time.sleep(1)
            
            data_json["MODE"] = "train" if data_json["MODE"] == "val" else "val"
            f.seek(0)
            json.dump(data_json, f, indent=4)
            f.truncate() 
