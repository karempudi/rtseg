
# Metrics are copied from Ryan Peters' torchvf library

import torch
import numpy as np

def f1(output, target):
    """ F1 or Dice score. """
    B, C, H, W = target.shape

    output = output.view(B, C, -1)
    target = target.view(B, C, -1)

    tp = torch.sum(output * target, dim = 2)
    fp = torch.sum(output, dim = 2) - tp
    fn = torch.sum(target, dim = 2) - tp

    # IoU Score
    score = (2 * tp) / ((2 * tp) + fp + fn)

    return score 

def iou(output, target):
    """ IoU or Jaccard score. """
    B, C, H, W = target.shape

    output = output.view(B, C, -1)
    target = target.view(B, C, -1)

    tp = torch.sum(output * target, dim = 2)
    fp = torch.sum(output, dim = 2) - tp
    fn = torch.sum(target, dim = 2) - tp

    # IoU Score
    score = tp / (tp + fp + fn)

    return score 


def compute_iou(y_pred, y_true):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        y_true (np.ndarray): Actual mask.
        y_pred (np.ndarray): Predicted mask.

    Returns:
        np.ndarray: IoU matrix, of size true_objects x pred_objects.

    """
    pred_objects = len(np.unique(y_pred))
    true_objects = len(np.unique(y_true))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        y_pred.flatten(), y_true.flatten(), 
        bins = (pred_objects, true_objects)

    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.histogram(y_true, bins = true_objects)[0]

    area_pred = np.expand_dims(area_pred, -1)
    area_true = np.expand_dims(area_true, 0)

    # Compute union
    union = area_pred + area_true - intersection

    # exclude background
    intersection = intersection[1:, 1:] 
    union        = union[1:, 1:]

    union[union == 0] = 1e-9

    iou = intersection / union
    
    return iou     


def precision_at(threshold, iou):
    matches = iou > threshold

    true_positives  = np.sum(matches, axis = 1) >= 1  # Correct
    false_positives = np.sum(matches, axis = 1) == 0  # Extra 
    false_negatives = np.sum(matches, axis = 0) == 0  # Missed

    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives)
    )

    return tp, fp, fn


def map_iou(y_preds, y_trues, threshold=np.arange(0.5, 1.0, 0.05)):
    """
    Computes mAP IOU, standard metric used to evaluate
    segmentation algorithms

    Masks contain segmented pixels, 0 label for background and
    1, 2, 3 ... labels for cell instances

    Args:
        y_preds (List[np.ndarray]): Predictions
        y_trues (List[np.ndarray]): Ground truths
    
    Returns:
        mAP (List[np.ndarray]): mAP at each threshold value

    """

    ious = [compute_iou(y_pred, y_true) for y_pred, y_true in zip(y_preds, y_trues)]

    mAP = [] # init to return a numpy array

    for t in threshold:
        tps, fps, fns = 0., 0., 0.
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn
        
        p = tps / (tps + fps + fns)
        mAP.append(p)
    
    return mAP, np.round(threshold, decimals=3).tolist()

