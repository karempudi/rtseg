
# File containing loss functions for YOLOv3 model
# Most of the code is from 
# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/loss.py
# Only modified very tiny bits to avoid passing the model and only pass anchors and strides
# to the loss functions

import torch
import torch.nn as nn
import math


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def compute_yolo_loss(predictions, targets, anchor_boxes, strides):
    
    device = targets.device
    ## placeholder variables for different losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    predictions_shapes = []
    for pred in predictions:
        predictions_shapes.append(pred.shape)
    # Build yolo targets
    tcls, tbox, indices, anch = build_targets(targets, anchor_boxes, strides, predictions_shapes)
    
    # define loss function
    # Loss function for classification and objectness
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    

    for layer_idx in range(len(predictions)):
        # Get image ids, anchors, grid index i, grid index j for each target in the yolo layer
        b, anchor, grid_j, grid_i = indices[layer_idx]
        # Build empty object target tensor with same shape as object prediction
        # tobj is one number per superpixel for each image and each anchor
        tobj = torch.zeros_like(predictions[layer_idx][..., 0], device=device)

        # Number of targets at this scale will be in b --> batch_indices for each image
        num_targets = b.shape[0]
        # Check if there are targets for this batch at this scale
        if num_targets:
            # perdictions at this scale
            ps = predictions[layer_idx][b, anchor, grid_j, grid_i]

            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multipy with anchor priors at this scale
            pwh = torch.exp(ps[:, 2:4]) * anch[layer_idx]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1)
            # Calcultae CIoU or GIoU for each target with the predicted bbox
            # tbox will be transposed in bbox_iou
            iou = bbox_iou(pbox.T, tbox[layer_idx], x1y1x2y2=False, CIoU=True)
            # We want to maximize iou and, max iou is 1 so we minimize (1 - IoU)
            lbox += (1.0 - iou).mean()

            # classification of objectness
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)

            # Classificatoin of class
            # check if we need to do classification
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device) # targets
                t[range(num_targets), tcls[layer_idx]] = 1
                # Calculate BCE loss
                lcls += BCEcls(ps[:, 5:], t) # BCE

        # Classification of objectness
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(predictions[layer_idx][..., 4], tobj) # obj loss
    
    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, torch.cat((lbox, lobj, lcls, loss)).cpu()


# basically morph the target into appropriate format for the loss function
# at each scale 
def build_targets(targets, anchors, strides, predictions_shapes):
    #
    n_anchors, n_targets = len(anchors), targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)
    # anchor_idx makes an index for each of the anchor box at each scale
    anchor_idx = torch.arange(n_anchors, device=targets.device).float().view(n_anchors, 1).repeat(1, n_targets)
    # repeat each target box for each anchor and then add the anchor index 0, 1, 2, ...for each  bbox
    # each bbox target get repeated 3 times with an index that corresponds to each anchor box at a 
    # particular scale
    targets = torch.cat((targets.repeat(n_anchors, 1, 1), anchor_idx[:, :, None]), 2)
    
    # loop for differet scales, each yolo layer corresponds to a scale with 
    # predictions of a certian shape,
    # Goal at each scale is to convert non-dimensional target-bboxes shapes to 
    # appropriate number at each scale for each superpixel
    
    for i in range(len(anchors)):
        
        # scale anchors to the yolo grid cell size so that an anchor with size of cell would result in 1
        anchors_i = anchors[i] / strides[i]
        # gain tensor is [1, 1, n_cells_x, n_cells_y, n_cells_x, n_cells_y, anchor_idx]
        gain[2: 6] = torch.tensor(predictions_shapes[i])[[3, 2, 3, 2]] # xyxy number of cells
        # Scale targets by the number of cells, they will be in yolo coordinate system
        t = targets * gain
        
        # if there are targets, check the ratios and if they are < 4 atleast in one direction
        if n_targets:
            # calucalate ration between anchor and target box for both width and height
            r = t[:, :, 4:6] / anchors_i[:, None]
            # Select ration that have the largest rations in any axis and check if it is less than 4
            j = torch.max(r, 1./r).max(2)[0] < 4 
            # Use targets that have ration 
            t = t[j]
        else:
            t = targets[0]
            
        # Extract image id in batch and class_id
        b, c = t[:, :2].long().T
        # x, y, h, w are already in cll coordinate system 
        gxy = t[:, 2:4] #xy
        gwh = t[:, 4:6] # target wh
        # Cast to int to get cell index e.g 1.2 gets associated to cell 1
        gij = gxy.long()
        # Isolate x and y index dimensions
        gi, gj = gij.T # grid xy indices
        # Convert anchor indices to ind
        a = t[:, 6].long()
        # Add target tensors for each layer 
        # Add to index lsit and limint index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, int(gain[3]) -1), gi.clamp_(0, int(gain[2]) - 1)))
        # Add to target box list and convert box coordinates from global grid corrdinate to local offsets
        # in the grid cell
        tbox.append(torch.cat((gxy - gij, gwh), 1)) 
        # add correct anchor for each target to the lsit
        anch.append(anchors_i[a])
        # Add class for each target to the list
        tcls.append(c)
    return tcls, tbox, indices, anch