FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV WORKSPACE=/workspace
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# Ubuntu 24.04 marks system Python as externally managed (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR ${WORKSPACE}

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-yaml \
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
    ros-jazzy-rclpy \
    ros-jazzy-std-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${WORKSPACE}/requirements.txt

# --ignore-installed: the apt numpy (pulled in by ROS) has no pip RECORD file, so pip
# cannot uninstall it. The pip copy in /usr/local takes precedence.
RUN python3 -m pip install --ignore-installed -r ${WORKSPACE}/requirements.txt && \
    python3 -m pip check

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
