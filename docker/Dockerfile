FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential\
    lsb-release

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gdb \
    git \
    gnupg \
    iproute2 \
    tmux \
    vim \
    wget \
    zlib1g-dev

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3 \
    libosmesa6-dev \
    libsdl2-dev \
    mesa-utils
# Necessary for mujoco
ENV LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so

ENV HOME /root
ENV PATH=$PATH:$HOME/.local/bin

# Install mujoco
RUN mkdir $HOME/.mujoco \
  && cd $HOME/.mujoco \
  && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
  && tar -zxvf mujoco210-linux-x86_64.tar.gz \
  && rm mujoco210-linux-x86_64.tar.gz \
  && cd $HOME
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
RUN pip install mujoco_py
RUN python -m mujoco_py || true

RUN pip install gym[all,accept-rom-license] \
                jupyter \
                scipy \
                tensorboard

# Install determined
#RUN pip install determined

ENV TORCH_HOME=/torch_home

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
