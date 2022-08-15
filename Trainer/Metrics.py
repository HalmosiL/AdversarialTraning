import torch
import numpy as np

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def acuracy(pred, label):
    pred = torch.nn.functional.relu(torch.sigmoid(pred))
    pred_sum = torch.sum(pred.clone(), dim=1)
    pred = torch.argmax(pred, dim=1)

    label = label.reshape(1, *label.shape)

    pred_sum = (torch.sign(pred_sum) - 1) * 200
    pred = torch.clip(pred + pred_sum, min=-1, max=20).int()

    return ((pred == label).sum() / (512 * 512)).item()