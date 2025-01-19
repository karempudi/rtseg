import torch
from rtseg.barcodedetect.networks import model_dict as barcode_model_dict
from rtseg.barcodedetect.utils import YoloLiveAugmentations, outputs_to_bboxes
from rtseg.barcodedetect.utils import non_max_suppression, YoloLiveUnAugmentations


def get_barcode_model(model_path, anchor_sizes, strides, num_classes=1, device='cuda:0'):

    barcode_model = barcode_model_dict['YOLOv3']

    anchors_list = [[anchor_sizes[i], anchor_sizes[i+1], anchor_sizes[i+2]] for i in range(0, len(anchor_sizes), 3)]
    anchors_t = tuple(torch.tensor(anch).float().to(device=device) for anch in anchors_list)
    strides_t = tuple(torch.tensor(stride).to(device=device) for stride in strides)

    barcode_model = barcode_model.parse(anchors=anchors_list, num_classes=num_classes).to(device=device)

    barcode_model.load_state_dict(torch.load(model_path, map_location=device))
    barcode_model.eval()

    return barcode_model, anchors_t, strides_t

def detect_barcodes(model, anchors, strides, image, model_img_size,
                        device='cuda:0', conf_thres=0.25, iou_thres=0.45):
    """
    Arguments:
        model is a pytorch model loaded with parmeters that you can use 
        to make predictions 

        anchors, strides: look 'get_barcode_model' for getting these two

        image: numpy image, float32

        model_img_size: imgsize on which barcode model is applied, used to get back to bboxes on 
                the original image
    """

    # transfrom first
    pre_barcode_transforms = YoloLiveAugmentations()
    barcode_sample = pre_barcode_transforms({'phase': image})

    # pass through the model
    barcode_pred = model(barcode_sample['phase'].unsqueeze(0).to(device))

    # compute bboxes and clean them up
    bboxes =  outputs_to_bboxes(barcode_pred, anchors, strides)
    bboxes_cleaned = non_max_suppression(bboxes, conf_thres=conf_thres, iou_thres=iou_thres)
    bboxes_barcode = [bbox.numpy() for bbox in bboxes_cleaned][0]

    # transform bboxes in the size of the original image
    yolo_datapoint = {
        'yolo_size': model_img_size,
        'bboxes': bboxes_barcode
    }

    raw_shape = image.shape

    post_barcode_transformations = YoloLiveUnAugmentations(
        parameters = {
            'resize': {
                'height': raw_shape[0],
                'width': raw_shape[1]
            }
        }
    )

    bboxes_final = post_barcode_transformations(yolo_datapoint)
    bboxes_final = sorted(bboxes_final, key=lambda x: x[0])

    return bboxes_final

# entry point into indentifying and mapping channels between two images
def channel_locations(image):
    pass
