import sys
import torch
from rtseg.cellseg.networks import model_dict
from rtseg.cellseg.utils.tiling import get_tiler
from rtseg.cellseg.numerics.vf_to_masks import construct_masks_batch
from rtseg.utils.hardware import get_device_str
from rtseg.oldseg.networks import model_dict as seg_model_dict
from rtseg.barcodedetect.networks import model_dict as barcode_model_dict
from rtseg.oldseg.transforms import UnetTestTransforms
from rtseg.barcodedetect.utils import YoloLiveAugmentations, YoloLiveUnAugmentations
from rtseg.barcodedetect.utils import non_max_suppression, outputs_to_bboxes
from skimage.morphology import remove_small_holes, remove_small_objects
#from rtseg.cellseg.utils.transforms import normalize99
from rtseg.identify_channels import get_channel_locations
from skimage.measure import label

def load_cellseg_model(model_path, device='cuda:0'):

    # TODO: hardcoding for now, do better later
    model = model_dict['ResUnet']
    model = model.parse(channels_by_scale=[1, 32, 64, 128, 256], num_outputs=[1, 2, 1],
                upsample_type='transpose_conv', feature_fusion_type='concat',
                skip_channel_seg=True)
    model.load_state_dict(torch.load(model_path))

    tiler = get_tiler("dynamic_overlap")
    wrapped_model = tiler(model, device=device)
    wrapped_model = wrapped_model.to(device)

    return wrapped_model

# Entry point to call segmentation of cells on an image
def cellsegment(model, image, device='cuda:0'):

    #image = normalize99(image).astype('float32')
    a = torch.from_numpy(image)[None, None, :].to('cuda:0') / 35000.0

    with torch.inference_mode():
        pred_semantic, pred_vf = model(a)
    mask = construct_masks_batch(pred_vf, pred_semantic, device=device, store_solutions=False, fast=True)
    
    return mask[0][0].numpy()


class LiveNet:

    def __init__(self, param):
        super().__init__()
        device = param.Hardware.device
        if torch.cuda.is_available():
            _, device_idx = get_device_str(device)
            if device_idx is not None:
                torch.cuda.set_device(device)
        else:
            device = 'cpu'
        
        self.device = device

        if 'Segmentation' in param.keys():
            # load the right type of segmentaiton network
            segment_params = param.Segmentation
            if segment_params.type == 'dual':
                # predicting 2 channels at the same time
                model = seg_model_dict[segment_params.architecture]
                self.segment_model = model.parse(channels_by_scale=segment_params.model_params.channels_by_scale,
                                                 num_outputs=segment_params.model_params.num_outputs,
                                                 upsample_type=segment_params.model_params.upsample_type,
                                                 feature_fusion_type=segment_params.model_params.feature_fusion_type).to(device=self.device)
        else:
            self.segment_model = None

        if 'BarcodeAndChannels' in param.keys():
            # load the right barcode network
            barcode_params = param.BarcodeAndChannels
            barcode_model = barcode_model_dict[barcode_params.architecture]
            anchor_sizes = barcode_params.model_params.anchors.sizes
            strides = barcode_params.model_params.anchors.strides
            num_classes = barcode_params.model_params.num_classes

            anchors_list = [[anchor_sizes[i], anchor_sizes[i+1], anchor_sizes[i+2]] for i in range(0, len(anchor_sizes), 3)]

            self.anchors_t = tuple(torch.tensor(anch).float().to(device=device) for anch in anchors_list)
            self.strides_t = tuple(torch.tensor(stride).to(device=device) for stride in strides)

            self.barcode_model = barcode_model.parse(anchors=anchors_list, num_classes=num_classes).to(device=self.device)
        else:
            self.barcode_model = None

        self.param = param

    def load_state_dict(self):
        
        segment_model_path = self.param.Segmentation.model_paths.both
        barcode_model_path = self.param.BarcodeAndChannels.model_path
        self.segment_model.load_state_dict(torch.load(segment_model_path, map_location=self.device))
        self.barcode_model.load_state_dict(torch.load(barcode_model_path, map_location=self.device))

    def eval(self):
        self.segment_model.eval()
        self.barcode_model.eval()

    def forward(self, x):
        return x # default


def get_live_model(param):

    net = LiveNet(param)
    net.load_state_dict()
    net.eval()
    return net


def live_segment(datapoint, model, param, visualize=False):
    """
    Arguments:
        datapoint: a dict with keys 'position', 'timepoint', 'phase', 'fluor'
            We always pass both images as we go through the queues as the later queues
            require you to hold onto the results from previous queues.
        model: an instance of LiveNet that is loaded with params and in eval mode
        param: a namespace of parameters used
        visualize: useful for plotting results into the UI

    """
    try:
        device = model.device

        if param.Segmentation.transformations.before_type == 'UnetTestTransforms':
            pre_segment_transforms = UnetTestTransforms() # make the phase image to the right shape with padding
        
        if param.BarcodeAndChannels.transformations.before_type == 'YoloLiveAugmentations':
            pre_barcode_transforms = YoloLiveAugmentations()

        raw_shape = datapoint['phase'].shape
        seg_sample = pre_segment_transforms({'phase': datapoint['phase'].astype('float32'),
                                            'raw_shape': raw_shape})
        barcode_sample = pre_barcode_transforms({'phase': datapoint['phase']})

        with torch.inference_mode():
            seg_pred = model.segment_model(seg_sample['phase'].unsqueeze(0).to(device)).sigmoid().cpu().numpy().squeeze(0)
            barcode_pred = model.barcode_model(barcode_sample['phase'].unsqueeze(0).to(device))
            bboxes  = outputs_to_bboxes(barcode_pred, model.anchors_t, model.strides_t)
            bboxes_cleaned = non_max_suppression(bboxes, conf_thres = param.BarcodeAndChannels.thresholds.conf,
                                                    iou_thres = param.BarcodeAndChannels.thresholds.iou)
            bboxes_barcode = [bbox.numpy() for bbox in bboxes_cleaned][0] # only one class so we should get this at index 0

        
        # now predictions are done
        yolo_img_size = tuple(param.BarcodeAndChannels.img_size)


        # cleaning up bbox predictions that are outside the size of the image
        # can happen as the net projects outward if the barcodes are at the edge
        # of the image
        for bbox in bboxes_barcode:
            if bbox[0] < 0.0:
                bbox[0] = 0.0
            if bbox[2] > yolo_img_size[1]:
                bbox[2] = yolo_img_size[1]
            if bbox[1] < 0.0:
                bbox[1] = 0.0
            if bbox[3] > yolo_img_size[0]:
                bbox[3] = yolo_img_size[0]

        # now you need to get the bboxs to the size of the original image        
        yolo_datapoint = {
            'yolo_size' : yolo_img_size,
            'bboxes': bboxes_barcode
        }

        post_barcode_transformations = YoloLiveUnAugmentations(
            parameters = {
                'resize' : {
                    'height': raw_shape[0],
                    'width': raw_shape[1],
                }
            }
        )

        bboxes_final = post_barcode_transformations(yolo_datapoint)
        bboxes_final = sorted(bboxes_final, key=lambda x: x[0]) # sort using the top left corner in x axis

        # Now figure out channel locations
        trap_locations, bboxes_taken = get_channel_locations(datapoint['phase'], bboxes_final,
                             param.BarcodeAndChannels.num_traps_per_block,
                             param.BarcodeAndChannels.distance_between_traps)
        num_traps = 0
        trap_locations_list = []
        for barcode_idx, left_of_barcode_traps in trap_locations.items():
            if len(left_of_barcode_traps) == param.BarcodeAndChannels.num_traps_per_block:
                trap_locations_list.extend(left_of_barcode_traps.tolist())
                num_traps += len(left_of_barcode_traps)

        

        cell_prob = param.Segmentation.thresholds.cells.probability

        # clean up the segmentation mask here if you want
        binary_cell_mask =  seg_pred[0][:raw_shape[0], :raw_shape[1]] > cell_prob
        binary_cell_mask = remove_small_holes(binary_cell_mask)
        binary_cell_mask = remove_small_objects(binary_cell_mask)

        seg_mask = label(binary_cell_mask)



        return {
            'phase': datapoint['phase'].astype('uint16'),
            'fluor': datapoint['fluor'],
            'position': datapoint['position'],
            'timepoint': datapoint['timepoint'],
            'barcode_locations': bboxes_final,
            'seg_mask': seg_mask,
            'trap_locations_list': trap_locations_list,
            'num_traps': num_traps,
            'error': False
        }

    except Exception as e:
        sys.stdout.write(f"Error {e} in process image function at position: {datapoint['position']} - time: {datapoint['timepoint']}\n")
        sys.stdout.flush()
        return {
            'phase': datapoint['phase'],
            'fluor': datapoint['fluor'],
            'position': datapoint['position'],
            'timepoint': datapoint['timepoint'],
            'num_traps': -1,
            'error': True
        }
