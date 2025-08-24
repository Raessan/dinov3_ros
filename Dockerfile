# syntax=docker/dockerfile:1.4

# Build arguments
ARG ROS2_DISTRIBUTION="jazzy"

# By default, use ROS 2 Humble as the base image
FROM osrf/ros:${ROS2_DISTRIBUTION}-desktop
ARG ROS2_DISTRIBUTION

WORKDIR /dinov3_ros
SHELL ["/bin/bash", "-c"]
COPY . /dinov3_ros

# # Install dependencies
RUN apt-get update &&  apt-get install -y python3 python3-pip
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa -y \
#     && apt-get update \
#     && apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv

# # Update python3 symlink to point to 3.11
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
# RUN update-alternatives --set python3 /usr/bin/python3.11

# # Ensure pip is for Python 3.11
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install requirements (except pytorch)
RUN pip3 install -r requirements.txt --ignore-installed --break-system-packages

# Install your pytorch version 
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129 --break-system-packages

# Install subpackage with the models and inference files
RUN pip install -e . --break-system-packages

# Install ROS 2 dependencies through rosdep
RUN source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash && \
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi && \
    apt-get update && \
    rosdep update -y && \
    rosdep install --from-paths . --ignore-src -r -y

# ---------------------- Packages build ------------------------------------
RUN export MAKEFLAGS="-j 4" && \
    echo "source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash" >> ~/.bashrc && \
    echo "source /dinov3_ros/ros2_ws/install/setup.bash" >> ~/.bashrc && \
    source /opt/ros/${ROS2_DISTRIBUTION}/setup.bash &&\
    cd ros2_ws &&\
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    . install/setup.bash


