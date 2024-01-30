# 这行代码是一个 Dockerfile 中的指令，它指示 Docker 使用 NVIDIA 的 CUDA 11.3.1 开发环境
# 和 Ubuntu 20.04 操作系统作为基础镜像。这意味着，当您构建 Docker 镜像时，Docker 将从
#  Docker Hub 下载名为 `nvidia/cuda:11.3.1-devel-ubuntu20.04` 的镜像，并将其用作您的镜像
#  的基础镜像。此外，该镜像还包含了一些其他的软件包，例如 `wget`、`build-essential`、`g++`、
#  `gcc`、`libgl1-mesa-glx`、`libglib2.0-0`、`openmpi-bin`、`openmpi-common`、
#  `libopenmpi-dev`、`libgtk2.0-dev` 和 `git`。这些软件包可能是为了支持某些特定的应用程序
#  或项目而安装的。¹²³
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0
