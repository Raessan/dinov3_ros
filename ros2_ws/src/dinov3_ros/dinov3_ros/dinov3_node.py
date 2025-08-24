import cv2
from typing import List, Dict
#from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
import numpy as np

from dinov3_toolkit.backbone.model_backbone import DinoBackbone
from dinov3_toolkit.head_detection.model_head import  DinoFCOSHead

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
        self.declare_parameter("input_image_topic", 'image/topic')

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

        # Modes
        self.declare_parameter("debug", False)
        self.declare_parameter("use_object_detection", False)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # General params
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
        self.patch_size = self.get_parameter("patch_size").get_parameter_value().integer_value
        self.img_mean = self.get_parameter("img_mean").get_parameter_value().double_array_value
        self.img_std = self.get_parameter("img_std").get_parameter_value().double_array_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.input_image_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value

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
        }

        # Modes
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.use_object_detection = self.get_parameter("use_object_detection").get_parameter_value().bool_value

        # Translate mean and std to 3D tensor
        self.img_mean = np.array(self.img_mean, dtype=np.float32)[:, None, None]
        self.img_std = np.array(self.img_std, dtype=np.float32)[:, None, None]

        # Open models
        try:
            self.dino_backbone_loader = torch.hub.load(
                repo_or_dir=self.dino_model["repo_path"],
                model=self.dino_model["model_name"],
                source=self.dino_model["source"],
                weights=self.dino_model["weights_path"]
            )
            self.dino_backbone = DinoBackbone(self.dino_backbone_loader, self.dino_model['n_layers']).to(self.device)
        except Exception as e:
            self.get_logger().error(f"Failed to load DINO model: {e}")
            return TransitionCallbackReturn.TRANSITION_FAILURE
        
        

        if self.use_object_detection:
            try:
                self.detection_head = DinoFCOSHead(backbone_out_channels=self.dino_model['embed_dim'], fpn_channels=self.detection_model['fpn_ch'], num_classes=self.detection_model['n_classes'], num_convs=self.detection_model['n_convs']).to(self.device)
                self.detection_head.load_state_dict(torch.load(self.detection_model['weights_path'], map_location = self.device))
            except Exception as e:
                self.get_logger().error(f"Failed to load detection model: {e}")
                return TransitionCallbackReturn.TRANSITION_FAILURE
            
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        return TransitionCallbackReturn.SUCCESS
    

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        return TransitionCallbackReturn.SUCCESS

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
