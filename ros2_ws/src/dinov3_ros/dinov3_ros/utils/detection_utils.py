# deps: rclpy, vision_msgs, std_msgs
import torch

from vision_msgs.msg import (
    Detection2DArray, Detection2D, BoundingBox2D, 
    ObjectHypothesisWithPose, Pose2D, Point2D
)

def outputs_to_detection2darray(boxes: torch.Tensor,
                        scores: torch.Tensor,
                        labels: torch.Tensor,
                        header) -> Detection2DArray:
    """
    boxes:  Nx4 tensor [x1, y1, x2, y2] in pixels
    scores: N   tensor in [0,1]
    labels: N   tensor (int class ids)
    header: std_msgs.msg.Header from the source image
    """
    # Move to CPU + plain Python types
    if boxes.is_cuda:  boxes = boxes.detach().cpu()
    if scores.is_cuda: scores = scores.detach().cpu()
    if labels.is_cuda: labels = labels.detach().cpu()

    boxes  = boxes.float()
    scores = scores.float()
    labels = labels.long()

    arr = Detection2DArray()
    arr.header = header

    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        det = Detection2D()
        det.header = header  # keep per-detection header consistent

        # Convert to center-size (axis-aligned; theta=0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w  = max(0.0, x2 - x1)
        h  = max(0.0, y2 - y1)

        bb = BoundingBox2D()
        bb.center = Pose2D()
        bb.center.position = Point2D()
        bb.center.position.x = float(cx)
        bb.center.position.y = float(cy)
        bb.center.theta = 0.0
        bb.size_x = float(w)
        bb.size_y = float(h)
        det.bbox = bb

        ohp = ObjectHypothesisWithPose()
        ohp.hypothesis.class_id = str(c)
        ohp.hypothesis.score = float(s)
        # ohp.pose left default (unused for 2D)

        det.results.append(ohp)
        arr.detections.append(det)

    return arr