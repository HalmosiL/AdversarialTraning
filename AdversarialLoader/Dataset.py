import os
import torch
import time
import json

class DatasetAdversarial:    
    def __init__(self, data_queue_path, len_dataset, slice_):
        self.len_dataset = len_dataset
        self.data_queue_path = data_queue_path
        self.slice_ = slice_

    def __getitem__(self, idx):
        image_ = None
        label_ = None

        count_no_data = 0
        
        path_a = int(idx / self.slice_) + 1
        path_b = idx % self.slice_
        
        count_no_data = 0
        image_path = self.data_queue_path + "image_" + str(path_a) + "_" + str(path_b) + "_.pt"
        label_path = self.data_queue_path + "label_" + str(path_a) + "_" + str(path_b) + "_.pt"

        remove_queue = []
        
        while(label is None):
            if(
                os.path.exists(image_path) and
                os.path.exists(label_path)
            ):
                try:
                    image_ = torch.load(image_path).clone()
                    label_ = torch.load(label_path).clone()
                    remove_queue.append([image_, label_])
                except Exception as e:
                    return []
            else:
                count_no_data += 1
                if(count_no_data > 1 and count_no_data % 200 == 0):
                    print("waiting for data sice:" + str(0.01 * count_no_data)[:5] + "(s)...", end="\r")

                time.sleep(0.01)

        return images, labels, remove_queue

    def __len__(self):
        return self.len_dataset
