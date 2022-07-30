import torch
import numpy as np

def iou_score_m(pred, target):
    ious = []

    pred = torch.nn.functional.relu(torch.sigmoid(pred) - 0.6)
    pred_sum = torch.sum(pred.clone(), dim=1)
    pred = torch.argmax(pred, dim=1)
    pred_sum = (torch.sign(pred_sum) - 1) * 200
    pred = torch.clip(pred + pred_sum, min=-1, max=20).int()

    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(-1, 20):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (torch.sign(pred_inds[target_inds])).long().sum().data.cpu().item()
        union = torch.sign(pred_inds[target_inds]).long().sum().data.cpu().item() + torch.sign(target_inds).long().sum().data.cpu().item() - intersection
        if union != 0:
            ious.append(float(intersection) / float(max(union, 1)))

    ious = np.array(ious)

    if(len(ious) == 0):
        return 0

    ious = ious.sum() / len(ious)

    return ious

def acuracy(pred, label):
    pred = torch.nn.functional.relu(torch.sigmoid(pred))
    pred_sum = torch.sum(pred.clone(), dim=1)
    pred = torch.argmax(pred, dim=1)

    label = label.reshape(1, *label.shape)

    pred_sum = (torch.sign(pred_sum) - 1) * 200
    pred = torch.clip(pred + pred_sum, min=-1, max=20).int()

    return ((pred == label).sum() / (512 * 512)).item()