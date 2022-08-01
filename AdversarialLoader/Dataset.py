import glob
import os
import torch
import time

class DatasetAdversarial:
    def __init__(self, data_queue_path, len_dataset, concatenate_number, plus_batch_num):
        self.len_dataset = len_dataset
        self.concatenate_number = concatenate_number
        self.data_queue_path = data_queue_path
        self.plus_batch_num = plus_batch_num

        self.data_in_queue = glob.glob(self.data_queue_path + "image_*")
        self.data_in_queue.sort(key=self.__sort__)

    def __sort__(self, k):
        return int(k.split("_")[-1].split(".")[0])

    def __getitem__(self, idx):
        self.data_in_queue = glob.glob(self.data_queue_path + "image_*")
        self.data_in_queue.sort(key=self.__sort__)

        images = []
        labels = []
        
        images_remove = []
        labels_remove = []

        if(idx + 1 == self.len_dataset and self.plus_batch_num != None):
            concatenate_number_actual = self.plus_batch_num
        else:
            concatenate_number_actual = self.concatenate_number

        while(len(images) < concatenate_number_actual):
            if(len(self.data_in_queue)):
                self.data_in_queue = glob.glob(self.data_queue_path + "image_*")
                self.data_in_queue.sort(key=self.__sort__)
                
                image_path = self.data_queue_path + self.data_in_queue[0].split("/")[-1]
                label_path = self.data_queue_path + "label" + self.data_in_queue[0].split("/")[-1][len("image"):]

                if(
                    os.path.exists(image_path) and
                    os.path.exists(label_path)
                ):
                    images.append(torch.load(image_path).clone())
                    labels.append(torch.load(label_path).clone())

                    images_remove.append(image_path)
                    labels_remove.append(label_path)
            else:
                time.sleep(0.5)
                self.data_in_queue = glob.glob(self.data_queue_path + "image_*")
                self.data_in_queue.sort(key=self.__sort__)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        
        for i range(len(images_remove)):
            os.remove(images_remove[i])
            os.remove(labels_remove[i])

        return images, labels

    def __len__(self):
        return self.len_dataset
