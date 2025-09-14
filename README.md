# dinov3_ros: DINOv3-based ROS 2 package for vision tasks

This repository provides ROS 2 nodes for performing multiple vision tasks—such as object detection, semantic segmentation, and depth estimation—using Meta’s [DINOv3](https://github.com/facebookresearch/dinov3) as the backbone. A key advantage of this approach is that the DINOv3 backbone features are computed only once (the most computationally demanding step), and these shared features are then reused by lightweight task-specific heads. This design significantly reduces redundant computation and makes multi-task inference more efficient.


## Table of Contents

1. [Installation](#installation)
2. [Docker](#docker)
3. [Usage](#usage)
4. [Tasks](#tasks)
5. [Demos](#demos)
6. [Licensing](#licensing)
7. [Bibliography](#bibliography)


## Installation

First, ROS2 Humble should be installed. Follow instructions for [ROS2 Humble installation](https://docs.ros.org/en/humble/Installation.html). Previous versions are not reliable due to the need of recent versions of Python to run DINOv3.

```
git clone https://github.com/Raessan/dinov3_ros.git
pip3 install -e .
cd ros2_ws 
colcon build
```

The only package that has to be installed separately is pytorch, due to its dependence with the CUDA version. For example:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129 
```

Finally, we provide weights for the lightweight heads developed by us, but the DINOv3 backbone weights should be requested and obtained from their [repo](https://github.com/facebookresearch/dinov3). Its default placement is in `dinov3_toolkit/backbone/weights`. The presented heads have been trained using the `vits16plus` model from DINOv3 as a backbone.

## Docker

If running with docker, two steps are needed to work: 

1. First install the [Nvidia Container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) in the host machine.

2. In the Dockerfile, change the pytorch version according to your CUDA version.

Then, run:

``` 
docker compose build
docker compose up
``` 

Any terminal should be opened as `docker exec -it dinov3_ros bash`.

## Usage

Launch the bringup file with `ros2 launch bringup dinov3_ros dinov3.launch.py arg1:=value arg2:=value`. The available launch arguments so far are:

- debug: Whether to publish debug images that help interpret visually the results of the tasks. For example, overlaid bounding boxes for the task of detection, or colored depth map in the task of depth estimation (default: *false*).

- perform_{*task*}: *task* can be any of the developed head (detection, segmentation, depth...) and this variables activates or deactivates the task (default: all *true*).

- topic_image: The name of the topic that contains the input image (default: *topic_image*).

- params_file: The path to the config file with required information for the models. This file is by default in `config/params.yaml` and contains important variables such as the `img_size` (default *640x640*, used to train the provided models), the `device` (default *cuda*) and the paths of the backbones and heads, along with variables to create the models or perform inference.

The file `params.yaml` should be changed before launching the bringup file if the variables should be different from the ones provided.

## Tasks

Each task has been trained in a separate repo to obtain a good precision using model heads with <5M parameters. We didn't try to beat SOTA models or perform an extensive research, so you are encouraged to bring a new lightweight model since we designed it to be plug and play: each task has a `head_{*task*}` subfolder inside `dinov3_toolkit` where we've put the `model_head.py` and an `utils.py` file that are copied from their original repo. Also, the `backbone` has a `model_backbone.py`, and there is a general `utils.py` with common functions in `dinov3_toolkit`. Also, notice that some tasks have a `class_names.txt` file with the names of all the classes that were used to train that particular task.

### Object detection

Check the following repo: [object_detection_dinov3](https://github.com/Raessan/object_detection_dinov3) (repo not public yet)

### Semantic segmentation

Check the following repo: [semantic_segmentation_dinov3](https://github.com/Raessan/semantic_segmentation_dinov3) (repo not public yet)

### Depth estimation

Check the following repo: [depth_dinov3](https://github.com/Raessan/depth_dinov3) (repo not public yet)

## Demo

<img src="assets/test_video_inference.gif" height="800">

## Licensing
- Code in this repo: Apache-2.0.
- DINOv3 submodule: licensed separately by Meta (see its LICENSE).
- We don't distribute DINO weights. Follow upstream instructions to obtain them.

## Bibliography

- [Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski (2025). Dinov3. *arXiv preprint arXiv:2508.10104.*](https://github.com/facebookresearch/dinov3)

- [González-Santamarta, Miguel Á (2023). yolo_ros](https://github.com/mgonzs13/yolo_ros) (used as reference for some part of the implementation)