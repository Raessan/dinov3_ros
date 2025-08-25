from dinov3_toolkit.head_detection.utils import decode_outputs

def detection_inference(model_detection, feats, img_tensor, score_thresh=0.2, nms_thresh=0.6):
    outputs = model_detection(feats)
    img_size = img_tensor.shape[2]
    first_stride = img_size / outputs['cls'][0].shape[2]
    strides = [first_stride]
    for l in range(1,len(outputs['cls'])):
        strides.append(first_stride*2**l)
    boxes, scores, labels = decode_outputs(outputs, img_tensor.shape[2:], strides, score_thresh=score_thresh, nms_thresh=nms_thresh)
    return boxes, scores, labels
    
