import glob
import sys
import torch
import torchvision.transforms as T
import os
import json
import numpy as np

sys.path.insert(0, "./")

from Model import get_DeepLabv3
from Metrics import iou_score_m, acuracy
from WBLogger import LogerWB

def cacheModel(cache_id, model, CONFIG):
    models = glob.glob(CONFIG["MODEL_CACHE"] + "*.pt")
    models.sort()
    torch.save(model.state_dict(), CONFIG["MODEL_CACHE"] + CONFIG["MODEL_NAME"] + str(cache_id) + ".pt")

    if len(models) > 5:
        os.remove(models[0])
        
    return cache_id + 1

def train(CONFIG_PATH, CONFIG, DEVICE, train_loader_adversarial, val_loader_adversarial, val_loader):
    if(DEVICE == "cuda:2"):
        DEVICE = "cuda:3"
    elif(DEVICE == "cuda:3"):
        DEVICE = "cuda:2"
    
    model = get_DeepLabv3(DEVICE, encoder_weights=None)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    lossFun = torch.nn.CrossEntropyLoss(ignore_index=-1)

    logger = LogerWB(CONFIG["WB_LOG"], print_messages=CONFIG["PRINT_LOG"])

    print("Traning started.....")

    cache_id = 0
    cache_id = cacheModel(cache_id, model, CONFIG)
    
    for e in range(CONFIG["EPOCHS"]):
        model = model.train()

        loss_train_epoch = 0
        iou_train_epoch = 0
        acc_train_epoch = 0

        batch_id = 1

        if(os.path.exists(CONFIG['DATA_QUEUE'])):
            for filename in glob.glob(CONFIG['DATA_QUEUE'] + "*.pt"):
                os.unlink(filename)
        
        cut_ = 0
        
        print("Train Adversarial loader length:", len(train_loader_adversarial))
        print("Val Adversarial loader length:", len(val_loader_adversarial))
        
        for data in train_loader_adversarial:
            if(len(data) == 3):
                image = data[0][0].to(DEVICE)
                label = data[1][0].to(DEVICE)
                remove_files = np.array(data[2]).flatten()

                optimizer.zero_grad()
                prediction = model(image)

                loss = lossFun(prediction, label)
                iou = iou_score_m(prediction, label)
                acc = acuracy(prediction, label) / CONFIG["TRAIN_BATCH_SIZE"]

                logger.log_loss_batch_train_adversarial(train_loader_adversarial.__len__(), e, batch_id, loss.item())
                logger.log_iou_batch_train_adversarial(train_loader_adversarial.__len__(), e, batch_id, iou)
                logger.log_acc_batch_train_adversarial(train_loader_adversarial.__len__(), e, batch_id, acc)

                iou_train_epoch += iou
                loss_train_epoch += loss.item()
                acc_train_epoch += acc

                loss.backward()
                optimizer.step()
                batch_id += 1

                if(e % CONFIG["MODEL_CACHE_PERIOD"] == 0):
                    cache_id = cacheModel(cache_id, model, CONFIG)

                for m in remove_files:
                    os.remove(m)
            else:
                print("Jump..")
                cut_ += 1

        loss_train_epoch = loss_train_epoch / (train_loader_adversarial.__len__() - cut_)
        iou_train_epoch = iou_train_epoch / (train_loader_adversarial.__len__() - cut_)
        acc_train_epoch = acc_train_epoch / (train_loader_adversarial.__len__() - cut_)

        logger.log_loss_epoch_train_adversarial(e, loss_train_epoch)
        logger.log_iou_epoch_train_adversarial(e, iou_train_epoch)
        logger.log_acc_epoch_train_adversarial(e, acc_train_epoch)

        torch.save(model.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + str(e) + ".pt")
        torch.save(optimizer.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + "_optimizer" + str(e) + ".pt")

        cache_id = cacheModel(cache_id, model, CONFIG)

        with open(CONFIG_PATH, 'r+') as f:
            data_json = json.load(f)
            data_json["MODE"] = "val"
            f.seek(0)
            json.dump(data_json, f, indent=4)
            f.truncate() 

        model = model.eval()

        loss_val_epoch = 0
        iou_val_epoch = 0
        acc_val_epoch = 0

        val_status = 0
        
        if(os.path.exists(CONFIG['DATA_QUEUE'] + "_val/")):
            for filename in glob.glob(CONFIG['DATA_QUEUE'] + "_val/*.pt"):
                os.unlink(filename)
        
        print("Val finished:" + str(val_status / val_loader_adversarial.__len__())[:5] + "%", end="\r")
        cut_ = 0
        for data in val_loader_adversarial:
            with torch.no_grad():
                if(len(data) == 3):
                    image_val = data[0][0].to(DEVICE)
                    label_val = data[1][0].to(DEVICE)
                    remove_files = np.array(data[2]).flatten()

                    prediction = model(image_val)
                    loss = lossFun(prediction, label_val)
                    iou = iou_score_m(prediction, label_val)
                    acc = acuracy(prediction, label_val) / CONFIG["TRAIN_BATCH_SIZE"]

                    iou_val_epoch += iou
                    loss_val_epoch += loss
                    acc_val_epoch += acc
                    val_status += 1
                    
                    print("Val finished:" + str(val_status / (val_loader_adversarial.__len__() - cut_))[:5] + "%", end="\r")
            
                    for m in remove_files:
                        os.remove(m)
                else:
                    print("jump...")
                    cut_ = cut_ + 1
                
        loss_val_epoch = loss_val_epoch / (val_loader_adversarial.__len__() - cut_)
        iou_val_epoch = iou_val_epoch / (val_loader_adversarial.__len__() - cut_)
        acc_val_epoch = acc_val_epoch / (val_loader_adversarial.__len__() - cut_)

        logger.log_loss_epoch_val(e, loss_val_epoch)
        logger.log_iou_epoch_val(e, iou_val_epoch)
        logger.log_acc_epoch_val(e, acc_val_epoch)

        with open(CONFIG_PATH, 'r+') as f:
            data_json = json.load(f)
            data_json["MODE"] = "train"
            f.seek(0)
            json.dump(data_json, f, indent=4)
            f.truncate() 
