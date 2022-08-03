from DataLoaderManager import DataLoaderManager

import os
import torch
import time
import json

class DatasetAdversarial:    
    def __init__(self, data_queue_path, len_dataset, concatenate_number, plus_batch_num, type_):
        self.len_dataset = len_dataset
        self.concatenate_number = concatenate_number
        self.data_queue_path = data_queue_path
        self.plus_batch_num = plus_batch_num
        self.dataLoaderManager = DataLoaderManager()
        self.type_ = type_

    def __getitem__(self, idx):
        if(idx + 1 == self.len_dataset and self.plus_batch_num != None):
            concatenate_number_actual = self.plus_batch_num
        else:
            concatenate_number_actual = self.concatenate_number

        i = 0
        
        images = []
        labels = []

        count_no_data = 0
        
        while(i < concatenate_number_actual):
            data = self.dataLoaderManager.getID(self.data_queue_path, self.type_)
            if(len(data)):
                count_no_data = 0
                image_path = data[0]
                label_path = data[1]

                images.append(torch.load(image_path).clone())
                labels.append(torch.load(label_path).clone())

                try:
                    os.remove(image_path)
                    os.remove(label_path)

                    i += 1
                except
                    print("Colison")
                    time.sleep(0.1)
            else:
                count_no_data += 1
                if(count_no_data > 1 and count_no_data % 20 == 0):
                    print("waiting for data sice:" + str(0.1 * count_no_data)[:5] + "(s)...", end="\r")

                time.sleep(0.1)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

        return images, labels

    def __len__(self):
        return self.len_dataset
