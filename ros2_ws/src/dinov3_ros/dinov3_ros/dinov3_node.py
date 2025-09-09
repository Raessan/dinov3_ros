import cv2
from typing import List, Dict
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from sensor_msgs.msg import Image

import torch
import numpy as np

from dinov3_toolkit.backbone.model_backbone import DinoBackbone

from dinov3_toolkit.utils import resize_transform, image_to_tensor

from dinov3_toolkit.head_detection.model_head import  DinoFCOSHead
from dinov3_toolkit.head_detection.inference import detection_inference
from dinov3_toolkit.head_detection.utils import DETECTION_CLASS_NAMES, img_with_detections

from dinov3_toolkit.head_segmentation.model_head import ASPPDecoder
from dinov3_toolkit.head_segmentation.model_head_light import Mask2FormerLiteHead
from dinov3_toolkit.head_segmentation.utils import generate_segmentation_overlay, outputs_to_maps

class Dinov3Node(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("dinov3_node")
        self.get_logger().info(f"[{self.get_name()}] Creating...")

        # General params
        self.declare_parameter("img_size", 800)
        self.declare_parameter("patch_size", 16)
        self.declare_parameter("img_mean", [0.485, 0.456, 0.406])
        self.declare_parameter("img_std", [0.229, 0.224, 0.225])
        self.declare_parameter("device", 'cuda')
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # DINO model params
        self.declare_parameter("dino_model.repo_path", '')
        self.declare_parameter("dino_model.model_name", 'dinov3_vits16plus')
        self.declare_parameter("dino_model.weights_path", '')
        self.declare_parameter("dino_model.n_layers", 12)
        self.declare_parameter("dino_model.embed_dim", 384)
        self.declare_parameter("dino_model.source", "local")

        # Object detection model params
        self.declare_parameter("detection_model.weights_path", '')
        self.declare_parameter("detection_model.fpn_ch", 192)
        self.declare_parameter("detection_model.n_convs", 4)
        self.declare_parameter("detection_model.n_classes", 80)
        self.declare_parameter("detection_model.score_thresh", 0.2)
        self.declare_parameter("detection_model.nms_thresh", 0.2)

        # Segmentation model params
        self.declare_parameter("segmentation_model.weights_path", '')
        self.declare_parameter("segmentation_model.classes_path", '')
        self.declare_parameter("segmentation_model.hidden_dim", 256)
        self.declare_parameter("segmentation_model.target_size", 320)

        # Modes
        self.declare_parameter("debug", False)
        self.declare_parameter("perform_detection", False)
        self.declare_parameter("perform_segmentation", False)

        self.get_logger().info(f"[{self.get_name()}] Created...")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        try:
            # General params
            self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
            self.patch_size = self.get_parameter("patch_size").get_parameter_value().integer_value
            self.img_mean = self.get_parameter("img_mean").get_parameter_value().double_array_value
            self.img_std = self.get_parameter("img_std").get_parameter_value().double_array_value
            self.device = self.get_parameter("device").get_parameter_value().string_value
            self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

            # DINO model params
            self.dino_model = {
                'repo_path': self.get_parameter('dino_model.repo_path').get_parameter_value().string_value,
                'model_name': self.get_parameter('dino_model.model_name').get_parameter_value().string_value,
                'weights_path': self.get_parameter('dino_model.weights_path').get_parameter_value().string_value,
                'n_layers': self.get_parameter('dino_model.n_layers').get_parameter_value().integer_value,
                'embed_dim': self.get_parameter('dino_model.embed_dim').get_parameter_value().integer_value,
                'source': self.get_parameter('dino_model.source').get_parameter_value().string_value,
            }

            # Object detection params
            self.detection_model = {
                'weights_path': self.get_parameter('detection_model.weights_path').get_parameter_value().string_value,
                'fpn_ch': self.get_parameter('detection_model.fpn_ch').get_parameter_value().integer_value,
                'n_convs': self.get_parameter('detection_model.n_convs').get_parameter_value().integer_value,
                'n_classes': self.get_parameter('detection_model.n_classes').get_parameter_value().integer_value,
                'score_thresh': self.get_parameter('detection_model.score_thresh').get_parameter_value().double_value,
                'nms_thresh': self.get_parameter('detection_model.nms_thresh').get_parameter_value().double_value,
            }

            # Semantic segmentation params
            self.segmentation_model = {
                'weights_path': self.get_parameter('segmentation_model.weights_path').get_parameter_value().string_value,
                'classes_path': self.get_parameter('segmentation_model.classes_path').get_parameter_value().string_value,
                'hidden_dim': self.get_parameter('segmentation_model.hidden_dim').get_parameter_value().integer_value,
                'target_size': self.get_parameter('segmentation_model.target_size').get_parameter_value().integer_value,
            }

            # Modes
            self.debug = self.get_parameter("debug").get_parameter_value().bool_value
            self.perform_detection = self.get_parameter("perform_detection").get_parameter_value().bool_value
            self.perform_segmentation = self.get_parameter("perform_segmentation").get_parameter_value().bool_value

            # Translate mean and std to 3D tensor
            self.img_mean = np.array(self.img_mean, dtype=np.float32)[:, None, None]
            self.img_std = np.array(self.img_std, dtype=np.float32)[:, None, None]

            # Image profile
            self.image_qos_profile = QoSProfile(
                reliability=self.reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )

            # Publishers
            self.pub_detection = self.create_lifecycle_publisher(Image, "detections", 10)
            self.pub_segmentation = self.create_lifecycle_publisher(Image, "segmentation_map", 10)

            # CV bridge
            self.cv_bridge = CvBridge()

     
        except Exception as e:
            self.get_logger().error(f"Configuration failed. Error: {e}")
            return TransitionCallbackReturn.FAILURE
            
        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")
        try:
            # Create subscription
            self.sub_image = self.create_subscription(Image, "topic_image", self.image_cb, self.image_qos_profile)

            # DINO model
            self.dino_backbone_loader = torch.hub.load(
                repo_or_dir=self.dino_model["repo_path"],
                model=self.dino_model["model_name"],
                source=self.dino_model["source"],
                weights=self.dino_model["weights_path"]
            )
            self.dino_backbone = DinoBackbone(self.dino_backbone_loader, self.dino_model['n_layers']).to(self.device)
            self.dino_backbone.eval()
        
            # Open detection model
            if self.perform_detection:
                self.detection_head = DinoFCOSHead(backbone_out_channels=self.dino_model['embed_dim'], fpn_channels=self.detection_model['fpn_ch'], num_classes=self.detection_model['n_classes'], num_convs=self.detection_model['n_convs']).to(self.device)
                self.detection_head.load_state_dict(torch.load(self.detection_model['weights_path'], map_location = self.device))
                self.detection_head.eval()

            # Open segmentation model
            if self.perform_segmentation:
                with open(self.segmentation_model["classes_path"]) as f:
                    self.segmentation_class_names = [line.strip() for line in f]
                segmentation_num_classes = len(self.segmentation_class_names)
                self.segmentation_head = ASPPDecoder(num_classes=segmentation_num_classes, in_ch=self.dino_model['embed_dim'], 
                                                  target_size=(self.segmentation_model["target_size"], self.segmentation_model["target_size"])).to(self.device)
                self.segmentation_head.load_state_dict(torch.load(self.segmentation_model['weights_path'], map_location = self.device))
                self.segmentation_head.eval()
        
        except Exception as e:
            self.get_logger().error(f"Activation failed. Error: {e}")
            return TransitionCallbackReturn.FAILURE
        
        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS
    

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS
    
    def image_cb(self, msg: Image) -> None:

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
        
        img_resized = resize_transform(cv_image, self.img_size, self.patch_size)
        img_tensor = image_to_tensor(img_resized, self.img_mean, self.img_std).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.dino_backbone(img_tensor)

        if self.perform_detection:
            boxes, scores, labels = detection_inference(self.detection_head, feats, img_tensor, 
                                                        score_thresh=self.detection_model['score_thresh'], 
                                                        nms_thresh=self.detection_model['nms_thresh'])
            img_with_boxes = img_with_detections(img_resized, boxes, scores, labels, class_names=DETECTION_CLASS_NAMES)

            # Convert to ROS Image message
            msg = self.cv_bridge.cv2_to_imgmsg(img_with_boxes, encoding='rgb8')

            # Publish
            self.pub_detection.publish(msg)

        if self.perform_segmentation:
            semantic_logits = self.segmentation_head(feats)

            semantic_map = outputs_to_maps(semantic_logits, (self.img_size, self.img_size))

            segmentation_img = generate_segmentation_overlay(
                img_resized,
                semantic_map,
                class_names=self.segmentation_class_names,
                alpha=0.6,
                background_index=0,
                seed=42,
                draw_semantic_labels=True, 
                semantic_label_fontsize=5,
            )

            # Convert to ROS Image message
            msg = self.cv_bridge.cv2_to_imgmsg(segmentation_img, encoding='rgb8')

            # Publish
            self.pub_segmentation.publish(msg)


        # results = self.yolo.predict(
        #     source=cv_image,
        #     verbose=False,
        #     stream=False,
        #     conf=self.threshold,
        #     iou=self.iou,
        #     imgsz=(self.imgsz_height, self.imgsz_width),
        #     half=self.half,
        #     max_det=self.max_det,
        #     augment=self.augment,
        #     agnostic_nms=self.agnostic_nms,
        #     retina_masks=self.retina_masks,
        #     device=self.device,
        # )
        

        # # publish detections
        # detections_msg.header = msg.header
        # self._pub.publish(detections_msg)

        # del results
        # del cv_image

def main():
    rclpy.init()
    node = Dinov3Node()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
