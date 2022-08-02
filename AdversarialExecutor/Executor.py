import torchvision.transforms as T
import glob
import time
import torch
import os
import copy
import math
import json
import sys

from Gen import run
from Cityscapes import CitySegmentation
from Adversarial import Cosine_PDG_Adam
from ModelLodaer import resnet_slice_model, get_resnet18_hourglass, load_model

class Executor:
    def __init__(
        self,
        config_name,
        model_cache,
        queue_size_train,
        queue_size_val, 
        data_queue, 
        data_path, 
        batch_size, 
        number_of_steps,
        data_set_start_index_train,
        data_set_end_index_train,
        data_set_start_index_val,
        data_set_end_index_val,
        device
    ):
        try: 
            print("Create data cache...")
            os.mkdir(data_queue)
            os.mkdir(data_queue[:-1] + "_val")
            print("Data cache created successfuly...")
        except OSError as error: 
            print("Data cache alredy exist...")  

        self.config_name = config_name
        self.model_cache = model_cache
        self.queue_size_train = queue_size_train
        self.queue_size_val = queue_size_val
        self.data_path = data_path
        self.model_name = None
        self.batch_size = batch_size
        self.device = device
        self.num_workers = 2
        self.number_of_steps = number_of_steps
        self.data_queue = data_queue

        self.data_set_start_index_train = data_set_start_index_train
        self.data_set_end_index_train = data_set_end_index_train
        self.data_set_start_index_val = data_set_start_index_val
        self.data_set_end_index_val = data_set_end_index_val
        
        input_transform = T.Compose([
            T.ToTensor(),
        ])

        self.train_data_set = CitySegmentation(
            root=self.data_path,
            split="train",
            transform=input_transform,
            start_index=data_set_start_index_train,
            end_index=data_set_end_index_train
        )

        self.val_data_set = CitySegmentation(
            root=self.data_path,
            split="val",
            transform=input_transform,
            start_index=data_set_start_index_val,
            end_index=data_set_end_index_val
        )

        self.train_data_set_loader = torch.utils.data.DataLoader(
            self.train_data_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.val_data_set_loader = torch.utils.data.DataLoader(
            self.val_data_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        if(data_set_end_index_train is None):
            self.train_data_set_len = math.ceil((train_data_set.__len__() - data_set_start_index_train) / self.batch_size)
        else:
            self.train_data_set_len = math.ceil((data_set_end_index_train - data_set_start_index_train) / self.batch_size)

        if(data_set_end_index_val is None):
            self.val_data_set_len = math.ceil((val_data_set.__len__() - data_set_start_index_val) / self.batch_size)
        else:
            self.val_data_set_len = math.ceil((data_set_end_index_val - data_set_start_index_val) / self.batch_size)

        self.train_element_id = 0
        self.val_element_id = 0

        self.attack = Cosine_PDG_Adam(
            step_size=1,
            clip_size=0.02
        )

    def start(self):
        train_iter = iter(self.train_data_set_loader)
        val_iter = iter(self.val_data_set_loader)

        train_id = 0
        val_id = 0

        while(True):
            config = json.load(open(sys.argv[1]))
            config_main = json.load(open("../Configs/config_main.json"))

            if(not config['Allow_TO_RUN']):
                print("Executor (ID_", str(config['ID']), ") is stoped...")
                break

            new_model_name = glob.glob(self.model_cache + "*.pt")

            if(not len(new_model_name)):
                if(self.model_name is None):
                    print("There is no model to use yet...")
                    time.sleep(2)
            else:
                new_model_name.sort()
                new_model_name = new_model_name[-1]              

                if(self.model_name != new_model_name):
                    print("Use model:", new_model_name)
                    self.model_name = new_model_name
  
                    model = resnet_slice_model(
                        load_model(new_model_name, self.device)
                    )

                if(config_main["MODE"] == "train"):
                    if(self.train_element_id < self.train_data_set_len):
                        number_elments_of_data_queue = len(glob.glob(self.data_queue + "/*"))

                        if(number_elments_of_data_queue < self.queue_size_train * 2):
                            if(self.train_element_id == 0):
                                print("Start generating traning data from:", self.data_set_start_index_train, " to:", self.data_set_end_index_train, "...")

                            try:
                                train_id += 1
                                self.train_element_id += 1

                                batch = next(train_iter)

                                run(
                                    id_=(train_id - 1) * config_main["NUMBER_OF_EXECUTORS"] + config["ID"],
                                    batch=batch,
                                    device=self.device,
                                    model=model,
                                    attack=self.attack,
                                    number_of_steps=self.number_of_steps,
                                    data_queue=self.data_queue
                                )
                                
                                with open('./train_queue.json', 'r+') as f:
                                    data = json.load(f)
                                    data['IDS'].append((train_id - 1) * config_main["NUMBER_OF_EXECUTORS"] + config["ID"])
                                    data['IDS'].sort()
                                    f.seek(0)
                                    json.dump(data, f)
                                    f.truncate()
                                
                            except StopIteration:
                                train_iter = iter(self.train_data_set_loader)
                        else:
                            print("Data queue(Train) is full process is waiting...")
                            time.sleep(1)

                        self.val_element_id = 0
                else:
                    if(self.val_data_set_len <= self.val_element_id):
                        print("Waiting(Train) for other executors to finish...")
                        time.sleep(1)

                if(config_main["MODE"] == "val"):
                    if(self.val_element_id < self.val_data_set_len):
                        if(self.val_element_id == 0):
                            print("Start generating val data from:", self.data_set_start_index_val, " to:", self.data_set_end_index_val, "...")

                        number_elments_of_data_queue = len(glob.glob(self.data_queue[:-1] + "_val" + "/*"))

                        if(number_elments_of_data_queue < self.queue_size_val * 2):
                            try:
                                val_id += 1
                                self.val_element_id += 1

                                batch = next(val_iter)

                                run(
                                    id_=(val_id - 1) * config_main["NUMBER_OF_EXECUTORS"] + config["ID"],
                                    batch=batch,
                                    device=self.device,
                                    model=model,
                                    attack=self.attack,
                                    number_of_steps=self.number_of_steps,
                                    data_queue=self.data_queue[:-1] + "_val/"
                                )
                            except StopIteration:
                                val_iter = iter(self.val_data_set_loader)
                        else:
                            print("Data queue(Val) is full process is waiting...")
                            time.sleep(1)
                    else:
                        self.train_element_id = 0
                else:
                    if(self.train_data_set_len <= self.train_element_id):
                        print("Waiting(VAL) for other executors to finish...")
                        time.sleep(1)



