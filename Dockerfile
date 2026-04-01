FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV WORKSPACE=/workspace
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR ${WORKSPACE}

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libusb-1.0-0 \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${WORKSPACE}/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r ${WORKSPACE}/requirements.txt

COPY apriltag_runner.py ${WORKSPACE}/apriltag_runner.py
COPY camera_calibration.py ${WORKSPACE}/camera_calibration.py
COPY check_container.py ${WORKSPACE}/check_container.py
COPY entrypoint.sh ${WORKSPACE}/entrypoint.sh

RUN chmod +x ${WORKSPACE}/entrypoint.sh

RUN mkdir -p \
    ${WORKSPACE}/calibration_images \
    ${WORKSPACE}/output \
    ${WORKSPACE}/tag_images

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]