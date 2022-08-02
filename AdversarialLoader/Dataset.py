import os
import torch
import time

class DatasetAdversarial:    
    def __init__(self, data_queue_path, len_dataset, concatenate_number, plus_batch_num):
        self.len_dataset = len_dataset
        self.concatenate_number = concatenate_number
        self.data_queue_path = data_queue_path
        self.plus_batch_num = plus_batch_num
        self.epoch = 0
        self.delete_q = []

    def __getitem__(self, idx):
        if(idx + 1 == self.len_dataset and self.plus_batch_num != None):
            concatenate_number_actual = self.plus_batch_num
        else:
            concatenate_number_actual = self.concatenate_number

        idx_ = idx * concatenate_number_actual + self.epoch * (self.len_dataset - 1)
        i = 0
        
        images = []
        labels = []
        
        images_remove = []
        labels_remove = []
        
        count_no_data = 0
        
        while(i < concatenate_number_actual):
            image_path = self.data_queue_path + "image_" + str(idx_ + i + 1) + ".pt"
            label_path = self.data_queue_path + "label_" + str(idx_ + i + 1) + ".pt"

            if(
                os.path.exists(image_path) and
                os.path.exists(label_path)
            ):
                count_no_data = 0
                images.append(torch.load(image_path).clone())
                labels.append(torch.load(label_path).clone())

                images_remove.append(image_path)
                labels_remove.append(label_path)

                i += 1
            else:
                count_no_data += 1
                if(count_no_data == 1):
                    print("waiting for data...")
                elif(count_no_data > 1):
                    print("waiting for data sice:" + str(0.05 * count_no_data) + "(s)...", end="\r")
                
                time.sleep(0.05)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        
        self.delete_q.append([images_remove, labels_remove])
        
        if(len(self.delete_q) > 5):
            for i in range(len(self.delete_q[0][0])):
                os.remove(self.delete_q[0][0][i])
                os.remove(self.delete_q[0][1][i])
                
            self.delete_q.pop(0)
            
        if(idx == self.len_dataset - 1):
            self.epoch += 1

        return images, labels

    def __len__(self):
        return self.len_dataset
