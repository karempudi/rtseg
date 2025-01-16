import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torchvision
import albumentations as A
import cv2
from torchvision import transforms

# Some code in the file is from UVA dlc course some is from 
# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/test.py


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def outputs_to_bboxes(outputs, anchors, strides):
    yolo_bboxes = []
    for i, (out, anch, stri) in enumerate(zip(outputs, anchors, strides)):
        #print(out.shape, anch.shape, stri.shape)
        # scale the outputs to bboxes correctly
        bs, _, ny, nx, no = out.shape
        grid = make_grid(nx, ny).to(out.device)
        anch = anch.view(1, -1, 1, 1, 2).to(out.device)
        out[..., 0:2] = (out[..., 0:2].sigmoid() + grid) * stri.to(out.device)
        out[..., 2:4] = torch.exp(out[..., 2:4]) * anch
        out[..., 4:] = out[..., 4:].sigmoid()
        out = out.view(bs, -1, no)
        yolo_bboxes.append(out)
    return torch.cat(yolo_bboxes, 1)


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def to_cpu(tensor):
    return tensor.detach().cpu()

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """
    Performs Non-Maximum Suppression (NMS) on inference results

    Args:
        prediction: 
        conf_thres:
        iou_thres: 
        classs: 
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def plot_results_batch(images_batch, outputs_batch):
    """
    Gives figures handles for plotting results to tensorboard

    Args:
        images_batch: numpy.ndarray (B, C, H, W), C=1 in our case
        outputs_batch: list of B elements, one element of each images bbox
                       Each element is list of bboxes (nboxes x 6) of format
                       (x1, y1, x2, y2, conf, class)
    Returns:
        fig_handles: a list of B fig handles, where each figure has bboxes
                     plotted on them appropriately 
    """
    # return a list of matplotlib figure objects
    fig_handles = []
    if (images_batch.shape[0] != len(outputs_batch)):
        return None
    for i in range(len(outputs_batch)):
        fig, ax = plt.subplots()
        ax.imshow(images_batch[i][0], cmap='gray')
        for row in outputs_batch[i]:
            rect = patches.Rectangle((row[0], row[1]), row[2] - row[0], row[3] - row[1], linewidth=1,
                                edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        fig_handles.append(fig)
    return fig_handles




class YoloLiveAugmentations:

    """
    Test time augmentations done on the image before the yolo net sees
    and image, basically doing image resizing and tensorizing

    Args:
        parameters: parameters used by various transformations.

    """

    def __init__(self, parameters = {
        'to_float' : {'max_value': 65535.0},
        'resize': {'height': 256, 'width': 800, 'interpolation': cv2.INTER_LINEAR, 'always_apply': True},
        'from_float': {'max_value': 65535.0}
    }):
        self.parameters = parameters
        self.transform = A.Compose([
            A.ToFloat(**parameters['to_float']),
            A.Resize(**parameters['resize']),
            A.FromFloat(**parameters['from_float'])
        ])
        self.img_size = (self.parameters['resize']['height'], self.parameters['resize']['width'])
    def __call__(self, datapoint):
        # datapoint is a dict with keys = {'phase', ...}
        datapoint['phase'] = cv2.cvtColor(datapoint['phase'], cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=datapoint['phase'])
        image = transforms.ToTensor()(transformed['image'][:, :, 0].astype('float32'))
        datapoint['phase'] = image
        return datapoint


class YoloLiveUnAugmentations:

    """
    Test time Unagumentatoins done on the  image after the yolo net sees the images and 
    gives the bounding boxes
    Args:
        parameters: parameters used by various transforms

    """
    def __init__(self, parameters = {
        'resize' : {'height': 816, 'width': 4096, 'always_apply':True} # this the resize height to ie, the original image
    }):
        self.transform = A.Compose([
            A.Resize(**parameters['resize'])
        ], bbox_params=A.BboxParams(format='pascal_voc')) #here we would have already converted the bboxes to pascal_voc
        
    def __call__(self, datapoint):
        # datapoint should have 'bboxes' and 'yolo_size'(size on which bboxes were calculated by the net)
        yolo_size = datapoint['yolo_size']
        transformed = self.transform(image = np.zeros(yolo_size), bboxes=datapoint['bboxes'])
        return transformed['bboxes']