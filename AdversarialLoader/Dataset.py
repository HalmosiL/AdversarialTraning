import os
import torch
import time
import json

class DatasetAdversarial:    
    def __init__(self, data_queue_path, len_dataset, concatenate_number, plus_batch_num, slice_):
        self.len_dataset = len_dataset
        self.concatenate_number = concatenate_number
        self.data_queue_path = data_queue_path
        self.plus_batch_num = plus_batch_num
        self.slice_ = slice_

    def __getitem__(self, idx):
        if(1 <= self.slice_):
            concatenate_number_actual = 1
        else:
            if(idx + 1 == self.len_dataset and self.plus_batch_num != None):
                concatenate_number_actual = self.plus_batch_num
            else:
                concatenate_number_actual = self.concatenate_number
            
        i = 0
        
        images = []
        labels = []
        remove_queue = []

        count_no_data = 0
        
        path_a = int(idx / self.slice_) + 1
        path_b = idx % self.slice_
        
        while(i < concatenate_number_actual):
            count_no_data = 0
            image_path = self.data_queue_path + "image_" + str(path_a) + "_" + str(path_b) + "_.pt"
            label_path = self.data_queue_path + "label_" + str(path_a) + "_" + str(path_b) + "_.pt"

            if(
                os.path.exists(image_path) and
                os.path.exists(label_path)
            ):
                try:
                    images.append(torch.load(image_path).clone())
                    labels.append(torch.load(label_path).clone())
                    remove_queue.append([image_path, label_path])

                    i += 1
                except Exception as e:
                    return []
            else:
                count_no_data += 1
                if(count_no_data > 1 and count_no_data % 200 == 0):
                    print("waiting for data sice:" + str(0.01 * count_no_data)[:5] + "(s)...", end="\r")

                time.sleep(0.01)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

        return images, labels, remove_queue

    def __len__(self):
        return self.len_dataset
