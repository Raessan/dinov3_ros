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
from vision_msgs.msg import Detection2DArray

import torch
import numpy as np

from dinov3_toolkit.backbone.model_backbone import DinoBackbone

from dinov3_toolkit.common import resize_transform, image_to_tensor

from dinov3_toolkit.head_detection.model_head import  DinoFCOSHead
from dinov3_toolkit.head_detection.utils import detection_inference, generate_detection_overlay

from dinov3_toolkit.head_segmentation.model_head import ASPPDecoder
from dinov3_toolkit.head_segmentation.utils import generate_segmentation_overlay, outputs_to_maps

from dinov3_toolkit.head_depth.model_head import DepthHeadLite
from dinov3_toolkit.head_depth.utils import depth_to_colormap

# Extra functions to convert data to msgs
from dinov3_ros.utils.detection_utils import outputs_to_detection2darray


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
        self.declare_parameter("detection_model.classes_path", '')
        self.declare_parameter("detection_model.fpn_ch", 192)
        self.declare_parameter("detection_model.n_convs", 4)
        self.declare_parameter("detection_model.score_thresh", 0.2)
        self.declare_parameter("detection_model.nms_thresh", 0.2)

        # Segmentation model params
        self.declare_parameter("segmentation_model.weights_path", '')
        self.declare_parameter("segmentation_model.classes_path", '')
        self.declare_parameter("segmentation_model.hidden_dim", 256)
        self.declare_parameter("segmentation_model.target_size", 320)

        # Depth model params
        self.declare_parameter("depth_model.weights_path", '')

        # Modes
        self.declare_parameter("debug", False)
        self.declare_parameter("perform_detection", False)
        self.declare_parameter("perform_segmentation", False)
        self.declare_parameter("perform_depth", False)

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
            self.image_reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

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
                'classes_path': self.get_parameter('detection_model.classes_path').get_parameter_value().string_value,
                'fpn_ch': self.get_parameter('detection_model.fpn_ch').get_parameter_value().integer_value,
                'n_convs': self.get_parameter('detection_model.n_convs').get_parameter_value().integer_value,
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

            # Depth params
            self.depth_model = {
                'weights_path': self.get_parameter('depth_model.weights_path').get_parameter_value().string_value,
            }

            # Modes
            self.debug = self.get_parameter("debug").get_parameter_value().bool_value
            self.perform_detection = self.get_parameter("perform_detection").get_parameter_value().bool_value
            self.perform_segmentation = self.get_parameter("perform_segmentation").get_parameter_value().bool_value
            self.perform_depth = self.get_parameter("perform_depth").get_parameter_value().bool_value

            # Translate mean and std to 3D tensor
            self.img_mean = np.array(self.img_mean, dtype=np.float32)[:, None, None]
            self.img_std = np.array(self.img_std, dtype=np.float32)[:, None, None]

            # Image profile
            self.image_qos_profile = QoSProfile(
                reliability=self.image_reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )

            # Publishers
            self.pubs = []
            self.pubs_debug = []

            self.pub_detections = self.make_pub(Detection2DArray, "dinov3/detections", 10)
            self.pub_sem_seg    = self.make_pub(Image,            "dinov3/sem_seg",    10)
            self.pub_depth      = self.make_pub(Image,            "dinov3/depth",      10)

            if self.debug:
                self.pub_img_detections = self.make_pub(Image, "dinov3/debug/img_detections", 10, debug=True)
                self.pub_img_sem_seg    = self.make_pub(Image, "dinov3/debug/img_sem_seg",    10, debug=True)
                self.pub_img_depth      = self.make_pub(Image, "dinov3/debug/img_depth",      10, debug=True)

            # CV bridge
            self.cv_bridge = CvBridge()

            # Counter image
            self.img_counter = 0

     
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

            # Activate publishers
            self.activate_pubs(state)

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
                with open(self.detection_model["classes_path"]) as f:
                    self.detection_class_names = [line.strip() for line in f]
                detection_num_classes = len(self.detection_class_names)
                self.detection_head = DinoFCOSHead(backbone_out_channels=self.dino_model['embed_dim'], 
                                                   fpn_channels=self.detection_model['fpn_ch'], 
                                                   num_classes=detection_num_classes, 
                                                   num_convs=self.detection_model['n_convs']).to(self.device)
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

            # Open depth model
            if self.perform_depth:
                self.depth_head = DepthHeadLite(in_ch=self.dino_model['embed_dim'], out_size=(self.segmentation_model["target_size"], self.segmentation_model["target_size"])).to(self.device)
                self.depth_head.load_state_dict(torch.load(self.depth_model['weights_path'], map_location = self.device))
                self.depth_head.eval()
        
        except Exception as e:
            self.get_logger().error(f"Activation failed. Error: {e}")
            return TransitionCallbackReturn.FAILURE
        
        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")
        self.destroy_subscription(self.sub_image)
        self.sub_image = None
        self.deactivate_pubs(state)
        if self.device=="cuda" and torch.cuda.is_available():
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()
        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        self.destroy_pubs()
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS
    

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        self.deactivate_pubs(state)
        self.destroy_pubs()
        if self.device=="cuda" and torch.cuda.is_available():
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS
    
    def image_cb(self, msg: Image) -> None:

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
        
        self.img_counter += 1
        
        img_resized = resize_transform(cv_image, self.img_size, self.patch_size)
        img_tensor = image_to_tensor(img_resized, self.img_mean, self.img_std).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.dino_backbone(img_tensor)

            if self.perform_detection:
                boxes, scores, labels = detection_inference(self.detection_head, feats, (self.img_size, self.img_size), 
                                                            score_thresh=self.detection_model['score_thresh'], 
                                                            nms_thresh=self.detection_model['nms_thresh'])
                
                detections_msg = outputs_to_detection2darray(boxes, scores, labels, msg.header)
                self.pub_detections.publish(detections_msg)

                
                # If debug is activated, we draw the image with detections and publish it
                if self.debug:
                    img_with_boxes = generate_detection_overlay(img_resized, boxes, scores, labels, class_names=self.detection_class_names)

                    # Convert to ROS Image message
                    img_detections_msg = self.cv_bridge.cv2_to_imgmsg(img_with_boxes, encoding='rgb8')

                    # Publish
                    self.pub_img_detections.publish(img_detections_msg)

            if self.perform_segmentation:
                semantic_logits = self.segmentation_head(feats)

                semantic_map = outputs_to_maps(semantic_logits, (self.img_size, self.img_size))

                sem = np.ascontiguousarray(semantic_map, dtype=np.uint16)  # HxW, class IDs
                sem_seg_msg = self.cv_bridge.cv2_to_imgmsg(sem, encoding='mono16')
                sem_seg_msg.header = msg.header

                self.pub_sem_seg.publish(sem_seg_msg)

                # If debug is activated, we obtain a colored instance map and publish it
                if self.debug:
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
                    img_sem_seg_msg = self.cv_bridge.cv2_to_imgmsg(segmentation_img, encoding='rgb8')

                    # Publish
                    self.pub_img_sem_seg.publish(img_sem_seg_msg)

            if self.perform_segmentation:
                depth_map = self.depth_head(feats)
                depth_np = np.ascontiguousarray(depth_map.squeeze().cpu().numpy(), dtype=np.float32)

                depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_np, encoding='32FC1')
                depth_msg.header = msg.header

                self.pub_depth.publish(depth_msg)

                # If debug is activated, we obtain a colored depth map and publish it
                if self.debug:
                    img_depth = depth_to_colormap(depth_np)
                    img_depth_msg = self.cv_bridge.cv2_to_imgmsg(img_depth, encoding='bgr8')
                    img_depth_msg.header = msg.header
                    self.pub_img_depth.publish(img_depth_msg)

    # Helper functions to create publishers
    def make_pub(self, msg_type, topic, depth=10, debug=False):
        p = self.create_lifecycle_publisher(msg_type, topic, depth)
        (self.pubs_debug if debug else self.pubs).append(p)
        return p

    def activate_pubs(self, state: LifecycleState):
        for p in (*self.pubs, *self.pubs_debug):
            try: p.on_activate(state)
            except Exception as e: self.get_logger().warn(f"Activate publisher failed: {e}")

    def deactivate_pubs(self, state: LifecycleState):
        for p in (*self.pubs, *self.pubs_debug):
            try: p.on_deactivate(state)
            except Exception as e: self.get_logger().warn(f"Deactivate publisher failed: {e}")

    def destroy_pubs(self):
        for p in (*self.pubs, *self.pubs_debug):
            try: self.destroy_publisher(p)
            except Exception: pass
        self.pubs.clear()
        self.pubs_debug.clear()

def main():
    rclpy.init()
    node = Dinov3Node()
    node.trigger_configure()
    # Here we only trigger the activate function. In a more complex scenario the changes of state can be managed differently
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
