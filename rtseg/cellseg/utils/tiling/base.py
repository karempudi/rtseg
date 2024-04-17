
# Tiling and batching code mostly copied form torchvf library
# small changes were done here and there.

import torch
import torch.nn as nn

from math import  ceil

def batch_of_n(arr, n):
    """ Given an array, split into batches of size n. """
    for i in range(0, len(arr), n): 
        yield arr[i : i + n]

class TilerBase(nn.Module):
    """
    Tiler will wrap the model and overload the forward 
    method to do batching and unbatching of tiles.

    The model is trained on size say (320 x 320) patches,
    if you want great performance on data that is similar to
    training data, you would want the images at inference time
    to be of the same resolution. So tiling can increase the 
    performance.

    Code of this tiling class is modified from torchvf package
    to accomodate inference on a batch of image of the same size.
    So you can call this class with (B, 1, H, W) and you will get
    vf (B, 2, H, W) and semantic (B, 1, H, W) as answers.

    Note: Don't use this class directly as this is missing a few
        functions that are implemented in it's subclasses.

    """
    def __init__(self, model, tile_size=(320, 320), overlap=48, batch_size=2,
                  device="cpu"):
        """
        Args:
            model (nn.Module) : a model that is loaded with trained parameters
            tile_size (int, int): tile size of the image to apply the model on
            overlap (int): tiles are constructed with some overlap between them
                        to avoid artifacts at the tile boundaries.
            batch_size (int): batch size of the tiles the model runs on in one
                        inference step
            device (str): 'cpu' or 'cuda:0' ...

        Returns:
            a sub class of model that is wrapped by the batch tiling mechanism
        """
        super(TilerBase, self).__init__()
        self.model = model
        self.TH, self.TW = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device

        self.stride_H = self.TH - self.overlap  # stride height
        self.stride_W = self.TW - self.overlap  # stride width

    def tile_transform(self, image, overflow):
        pass
        
    def tile_image(self, image):
        B, C, H, W = image.shape

        # strict enforcement as images in test sets are different shape
        # For inference at run time, we can use batching of tiles from a 
        # batch of images of the same shape.
        #if self.strict:
        #    assert B == 1, f"Batch size of 1 required in strict mode. Found batch size {B}."

        overflow = self._tiling_overflow(image.shape)
        # all subclasses of tiler should implement a tile_transfrom
        padded_image = self.tile_transform(image, overflow)

        B, C, NH, NW = padded_image.shape

        # Now if you check tiling overflow, it should return 0s
        H_check, W_check = self._tiling_overflow((B, C, NH, NW))

        assert not H_check, "Tiling failed in the height dimension"
        assert not W_check, "Tiling failed in the width dimension"

        # 1. Find the number of tiles in X and Y direction
        H_tiles = int(1 + (NH - self.TH) / (self.TH - self.overlap))
        W_tiles = int(1 + (NW - self.TW) / (self.TW - self.overlap))
        
        # 2. The actual indices of tiles: (n_tiles, 4)
        tiles = []
        for H_n in range(H_tiles):
            for W_n in range(W_tiles):
                left = int(H_n * (self.TH - self.overlap))
                right = int(W_n * (self.TW - self.overlap))

                # Going to be from (y_0, x_0, y_1, x_1)
                tiles.append([left, right, left + self.TH, right + self.TW])
        #print(f"Tiles: {tiles}")
        # 3.
        tiled_images = []
        for i in range(B):
            for y_0, x_0, y_1, x_1 in tiles:
                tiled_images.append(
                    padded_image[i][:, y_0:y_1, x_0:x_1]
                )
            
        tiled_images = torch.stack(tiled_images)
        
        return tiled_images, tiles, padded_image.shape

    def _tiling_overflow(self, image_size):
        #print(image_size)
        
        B, C, H, W = image_size

        H_overflow = self.TH + self.stride_H * ceil(max((H - self.TH), 0) / self.stride_H) - H
        W_overflow = self.TW + self.stride_W * ceil(max((W - self.TW), 0) / self.stride_W) - W

        #print(f"H_overflow: {H_overflow}, W_overflow: {W_overflow}")
        return H_overflow, W_overflow

    def merge_transform(self):
        pass
        
    def merge_tiles(self, tiled_images, tiles, new_shape, original_shape):
        B, _, H, W = original_shape
        #print(f"Original batch size: {B}")
        _, C, TH, TW = tiled_images.shape
        _, _, NH, NW = new_shape

        # Don't want to include borders when stitching back
        # together as they often generate artifacts
        extra = int(self.overlap // 4) # 25 % of the overlap

        average = torch.zeros((B, C, NH, NW), device=self.device)
        merged_pred = torch.zeros((B, C, NH, NW), device=self.device)
        
        n_tiles = len(tiles)
        for i, (y_0, x_0, y_1, x_1) in enumerate(tiles):
            tiled_image = tiled_images[i::n_tiles, ...]
            #print(f"Tiled image shape: {tiled_image.shape}, [y_0, x_0, y_1, x_1]: {y_0}, {x_0}, {y_1}, {x_1}")
            ty_0, tx_0, ty_1, tx_1 = 0, 0, self.TH, self.TW

            if y_0:
                y_0 = y_0 + extra
                ty_0 = extra
                
            if x_0:
                x_0 = x_0 + extra
                tx_0 = extra

            if y_1 != NH:
                y_1 = y_1 - extra
                ty_1 = self.TH - extra

            if x_1 != NW:
                x_1 = x_1 - extra
                tx_1 = self.TW - extra
            #print(f"[y_0, x_0, y_1, x_1]: {y_0}, {x_0}, {y_1}, {x_1}")
            #print(f"[ty_0, tx_0, ty_1, tx_1]: {ty_0}, {tx_0}, {ty_1}, {tx_1}")
            average[:, :, y_0:y_1, x_0:x_1] += 1
            merged_pred[:, :, y_0:y_1, x_0:x_1] += tiled_image[:, :, ty_0: ty_1, tx_0: tx_1]

        merged_pred = merged_pred / average

        # For paddig and reshape this will reshape final predictions
        pred = self.merge_transform(merged_pred, new_shape, original_shape)

        return pred
    
    def forward(self, image):
        tiled_images, tiles, new_shape = self.tile_image(image)
        #return None
        batches = batch_of_n(tiled_images, self.batch_size)
        
        # Predict VF and semantic on each tiling batch
        vf_batched = []
        semantic_batched = []
        for batch in batches:
            semantic, vf = self.model(batch)
            
            semantic_batched.append(semantic)
            vf_batched.append(vf)

        vf_batched = torch.vstack(vf_batched)
        semantic_batched = torch.vstack(semantic_batched)

        #print(f"Vf_batched: {vf_batched.shape}")
        #print(f"Semantic batched: {semantic_batched.shape}")
        #print(f"Tiles in forward just before merging: {tiles}")
        # Stitching of predicted tiles
        vf = self.merge_tiles(vf_batched, tiles, new_shape, image.shape)
        semantic = self.merge_tiles(semantic_batched, tiles, new_shape, image.shape)

        #print("After merging")
        #print(f"vf shape: {vf.shape}")
        #print(f"semantic shape: {semantic.shape}")
        #return None
        return semantic, vf