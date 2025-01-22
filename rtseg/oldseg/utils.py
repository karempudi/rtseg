
import numpy as np
import torch.optim as optim
from skimage.measure import label
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from skimage import io
import scipy.ndimage as ndi


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

def generate_weights(filename, sigma=5, w0=10):

    img = io.imread(filename)
    # removing objects and calculating distances to objects needs labelled images
    labeledImg, num = label(img, return_num=True, connectivity=2)
    # remove small objects
    labeledImg = remove_small_objects(labeledImg, min_size=250)
    # unique values == number of blobs
    unique_values = np.unique(labeledImg) 
    num_values = len(unique_values)
    h, w = labeledImg.shape
    # stack keeps distance maps each blob
    stack = np.zeros(shape=(num_values, h, w))
    for i in range(num_values):
        stack[i] = ndi.distance_transform_edt(~(labeledImg == unique_values[i]))
    # sort the distance
    sorted_distance_stack = np.sort(stack, axis=0)
    # d1 and d2 are the shortest and second shortest distances to each object, 
    # sorted_distance_stack[0] is distance to the background. One can ignore it
    distance_sum = sorted_distance_stack[1] + sorted_distance_stack[2]
    squared_distance = distance_sum ** 2/ (2 * (sigma**2))
    weightmap = w0 * np.exp(-squared_distance)*(labeledImg == 0)
    return weightmap


def to_cpu(tensor):
    return tensor.detach().cpu()

def plot_results_batch(phase_batch, predictions_batch):
    """
    Gives figures handles for plotting results to tensorboard

    Args:
        phase_batch: numpy.ndarray (B, C, H, W), C=1 in our case
        predictions_batch: numpy.ndarray(B, C, H, W), C = 1 or 2 for Unet, more for omni
        Returns:
        fig_handles: a list of B fig handles, where each figure has bboxes
                     plotted on them appropriately 
    """
    # return a list of matplotlib figure objects
    fig_handles = []
    B, n_outputs, _, _ = predictions_batch.shape

    for i in range(B):
        fig, ax = plt.subplots(nrows=1 + n_outputs, ncols=1)
        fig.tight_layout()
        ax[0].imshow(phase_batch[i][0], cmap='gray')
        for j in range(n_outputs):
            ax[1 + j].imshow(predictions_batch[i][j], cmap='gray')
        fig_handles.append(fig)
    return fig_handles