
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetDualLoss(nn.Module):
    """
    Loss function when both cell mask and channel mask are used
    weights are optional

    """

    def __init__(self):
        super(UnetDualLoss, self).__init__()
        self.bce_loss_ch = nn.BCELoss()

    def forward(self, predictions, mask, channel_mask, weights=None):
        predictions = torch.sigmoid(predictions)

        batch_size, n_outputs, _, _ = predictions.shape
        assert n_outputs == 2, "Number of outputs is not 2, so not a dual network, use different loss function"


        predictions = predictions.view(batch_size, n_outputs, -1)
        predictions_cell_mask = predictions[:, 0, :]
        predictions_channel_mask = predictions[:, 1, :]

        target_cell_mask = mask.view(batch_size, -1) 
        target_channel_mask = channel_mask.view(batch_size, -1)

        # iou loss cells
        cell_intersection = (predictions_cell_mask * target_cell_mask)
        dice_per_image = 2. * (cell_intersection.sum(1)) / (predictions_cell_mask.sum(1) + target_cell_mask.sum(1))
        dice_batch_loss = 1. - dice_per_image.sum() / batch_size

        # iou loss channels
        channel_intersection = (predictions_channel_mask * target_channel_mask)
        dice_per_image_ch = 2. * (channel_intersection.sum(1)) / (predictions_channel_mask.sum(1) + target_channel_mask.sum(1))
        dice_batch_loss_ch = 1. - (dice_per_image_ch.sum()) / batch_size

        # cells weighted cross entropy
        if weights is not None:
            weights_reshaped = weights.view(batch_size, -1) + 1.0
            bce_loss = weights_reshaped * F.binary_cross_entropy(predictions_cell_mask, target_cell_mask, reduction='none')
            bce_loss_cells = bce_loss.mean()

            loss_cells = dice_batch_loss + 0.5 * bce_loss_cells
        else: # no weights normal cross entropy
            bce_loss = F.binary_cross_entropy(predictions_cell_mask, target_cell_mask, reduction='none')
            bce_loss_cells = bce_loss.mean()

            loss_cells = dice_batch_loss + 0.5 * bce_loss_cells

        # channels weighted cross entropy
        bce_loss_ch = self.bce_loss_ch(predictions_channel_mask, target_channel_mask)
        loss_ch = dice_batch_loss_ch + 0.5 * bce_loss_ch

        # cells shouldn't have anything outside the channels
        inverted_channel_mask = 1. - predictions_channel_mask
        cells_outside_ch_loss = torch.sum(((predictions_cell_mask - target_cell_mask) * inverted_channel_mask)**2.0) / torch.sum(inverted_channel_mask)

        #print(f"Cells: dice: {dice_batch_loss.item()} - bce: {bce_loss_cells.item()}")
        #print(f"Channels: dice: {dice_batch_loss_ch.item()} - bce: {bce_loss_ch.item()}")
        #print(f"Cell loss: {loss_cells.item()} -- channel loss: {loss_ch.item()}")
        #print(f"Cells outside channels: {cell_outside_ch_loss.item()}")
        loss_parts_dict = {
            'cell_bce_loss': bce_loss_cells.item(),
            'channel_bce_loss': bce_loss_ch.item(),
            'cells_dice_loss': dice_batch_loss.item(),
            'channels_dice_loss': dice_batch_loss_ch.item(),
            'loss_cells': loss_cells.item(),
            'loss_ch': loss_ch.item(),
            'outside_channel_loss': cells_outside_ch_loss.item()
        }

        return (2.0* loss_cells + loss_ch + cells_outside_ch_loss), loss_parts_dict
