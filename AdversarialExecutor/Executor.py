import glob
import time
import torch
import os
import math
import json
import sys

from Gen import run
from Adversarial import Cosine_PDG_Adam
import Transforms as transform
from Dataset import SemData
from ModelLodaer import slice_model, load_model_slice

class Executor:
    def sort_(self, key):
        key = key.split("_")[-1]
        key = key.split(".")[0]

        return int(key)
    
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
        device,
        num_workers,
        train_batch_size
    ):
        self.mode = None
        
        try: 
            print("Create data cache...")
            os.mkdir(data_queue)
            os.mkdir(data_queue[:-1] + "_val")
            print("Data cache created successfuly...")
        except OSError as error: 
            print("Data cache alredy exist...")  

        self.split = -1
        self.split_size = 0
            
        if(train_batch_size < batch_size):
            if(batch_size % train_batch_size != 0):
                print("batch_size", batch_size)
                raise ValueError(
                    "The executor batch size should be divisible by the train batch size...." +
                    "\ntrain_batch_size" + str(train_batch_size) +
                    "\nbatch_size" + str(batch_size))
            else:
                self.split = int(batch_size / train_batch_size)
                self.split_size = int(batch_size / self.split)
            
        self.config_name = config_name
        self.model_cache = model_cache
        self.queue_size_train = queue_size_train
        self.queue_size_val = queue_size_val
        self.data_path = data_path
        self.model_name = None
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.number_of_steps = number_of_steps
        self.data_queue = data_queue

        self.data_set_start_index_train = data_set_start_index_train
        self.data_set_end_index_train = data_set_end_index_train
        self.data_set_start_index_val = data_set_start_index_val
        self.data_set_end_index_val = data_set_end_index_val

        config_main = json.load(open("../Configs/config_main.json"))
        args = config_main['DATASET']

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        train_transform = transform.Compose([
            transform.RandScale([args["scale_min"], args["scale_max"]]),
            transform.RandRotate([args["rotate_min"], args["rotate_max"]], padding=mean, ignore_label=args["ignore_label"]),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args["train_h"], args["train_w"]], crop_type='rand', padding=mean, ignore_label=args["ignore_label"]),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        train_data = SemData(
            split='train',
            data_root=data_path,
            data_list=args["train_list"],
            transform=train_transform
        )

        val_transform = transform.Compose([
            transform.Crop([args["train_h"], args["train_w"]], crop_type='center', padding=mean, ignore_label=args["ignore_label"]),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        val_data = SemData(
            split='val',
            data_root=data_path,
            data_list=args["val_list"],
            transform=val_transform
        )

        self.train_data_set_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False
        )

        self.val_data_set_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )

        if(data_set_end_index_train is None):
            self.train_data_set_len = int((train_data_set.__len__() - data_set_start_index_train) / self.batch_size)
        else:
            self.train_data_set_len = int((data_set_end_index_train - data_set_start_index_train) / self.batch_size)

        if(data_set_end_index_val is None):
            self.val_data_set_len = math.ceil((val_data_set.__len__() - data_set_start_index_val) / self.batch_size)
        else:
            self.val_data_set_len = math.ceil((data_set_end_index_val - data_set_start_index_val) / self.batch_size)

        self.train_element_id = 0
        self.val_element_id = 0

        self.attack = Cosine_PDG_Adam(
            step_size=1,
            clip_size=0.03
        )

    def start(self):
        train_iter = iter(self.train_data_set_loader)
        val_iter = iter(self.val_data_set_loader)

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
                new_model_name.sort(key=self.sort_)
                new_model_name = new_model_name[-1]              

                if(self.model_name != new_model_name):
                    print("Use model:", new_model_name)
                    self.model_name = new_model_name
  
                    model = load_model_slice(new_model_name, self.device)

                if(config_main["MODE"] == "train"):
                    if(self.mode != "train"):
                        self.mode = "train"
                        self.train_element_id = 0
                        train_iter = iter(self.train_data_set_loader)
            
                    if(self.train_element_id < self.train_data_set_len):
                        number_elments_of_data_queue = len(glob.glob(self.data_queue + "/*"))

                        if(number_elments_of_data_queue < self.queue_size_train * 2):
                            if(self.train_element_id == 0):
                                print("Start generating traning data from:", self.data_set_start_index_train, " to:", self.data_set_end_index_train, "...")

                            try:
                                self.train_element_id += 1

                                batch = next(train_iter)
                                    
                                run(
                                    id_=(self.train_element_id - 1) * config_main["NUMBER_OF_EXECUTORS"] + config["ID"],
                                    batch=batch,
                                    device=self.device,
                                    model=model,
                                    attack=self.attack,
                                    number_of_steps=self.number_of_steps,
                                    data_queue=self.data_queue,
                                    split=self.split,
                                    split_size=self.split_size
                                )                               
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
                    if(self.mode != "val"):
                        self.mode = "val"
                        self.train_element_id = 0
                        val_iter = iter(self.val_data_set_loader)
                    
                    if(self.val_element_id < self.val_data_set_len):
                        if(self.val_element_id == 0):
                            print("Start generating val data from:", self.data_set_start_index_val, " to:", self.data_set_end_index_val, "...")

                        number_elments_of_data_queue = len(glob.glob(self.data_queue[:-1] + "_val" + "/*"))

                        if(number_elments_of_data_queue < self.queue_size_val * 2):
                            try:
                                self.val_element_id += 1

                                batch = next(val_iter)

                                run(
                                    id_=(self.val_element_id - 1) * config_main["NUMBER_OF_EXECUTORS"] + config["ID"],
                                    batch=batch,
                                    device=self.device,
                                    model=model,
                                    attack=self.attack,
                                    number_of_steps=self.number_of_steps,
                                    data_queue=self.data_queue[:-1] + "_val/",
                                    split=self.split,
                                    split_size=self.split_size
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



