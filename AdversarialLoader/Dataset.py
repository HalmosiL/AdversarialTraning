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
        
        if(type_ == "train"):
            self.path_queue = "../AdversarialExecutor/train_queue.json"
        else:
            self.path_queue = "../AdversarialExecutor/val_queue.json"

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
            idx_ = None
            
            with open(self.path_queue, 'r+') as f:
                data = json.load(f)
                if(len(data['IDS']) != 0):
                    idx_ = data['IDS'][0]
                    data['IDS'].pop(0)
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
            
            image_path = self.data_queue_path + "image_" + str(idx_) + ".pt"
            label_path = self.data_queue_path + "label_" + str(idx_) + ".pt"

            if(
                idx_ is not None and
                os.path.exists(image_path) and
                os.path.exists(label_path)
            ):
                count_no_data = 0
                images.append(torch.load(image_path).clone())
                labels.append(torch.load(label_path).clone())
                
                os.remove(image_path[-1])
                os.remove(labels[-1])

                i += 1
            else:
                count_no_data += 1
                if(count_no_data == 1):
                    print("waiting for data...\n")
                elif(count_no_data > 1):
                    print("waiting for data sice:" + str(0.05 * count_no_data)[:5] + "(s)...", end="\r")
                
                time.sleep(0.05)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

        return images, labels

    def __len__(self):
        return self.len_dataset
