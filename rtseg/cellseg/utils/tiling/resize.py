
from rtseg.cellseg.utils.tiling.base import TilerBase
import torchvision # type: ignore

class ResizeTiler(TilerBase):
    def tile_transform(self, image, overflow):
        B, C, H, W = image.shape
        H_overflow, W_overflow = overflow

        H_resize = H + H_overflow
        W_resize = W + W_overflow

        resized_image = torchvision.transforms.Resize((H_resize, W_resize), antialias=True)(image)

        return resized_image
        
    def merge_transform(self, merged_pred, new_shape, original_shape):
        _, _, H, W = original_shape

        pred = torchvision.transforms.Resize((H, W), antialias=True)(merged_pred)

        return pred