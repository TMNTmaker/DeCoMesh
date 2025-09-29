# ベースイメージ（CUDA 12.1 + Ubuntu 22.04）
#FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
#FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
#LABEL name="unique3d" maintainer="unique3d"

# 作業ディレクトリの作成
RUN mkdir -p /workspace
WORKDIR /workspace

# システムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    vim \
    curl \
    unzip \
    git-lfs \
    pkg-config \
    cmake \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libopenblas-dev \
    mesa-utils-extra \
    libeigen3-dev \
    libblas-dev\
    python3-pip git nano \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV QT_QPA_PLATFORM=offscreen
# 環境変数設定
#ENV PYTHONDONTWRITEBYTECODE=1
#ENV PYTHONUNBUFFERED=1
#ENV PYOPENGL_PLATFORM=egl
#ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# ===============================
# Miniconda のインストール
# ===============================
#ENV CONDA_DIR=/opt/conda
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
#    rm Miniconda3-latest-Linux-x86_64.sh
#ENV PATH=$CONDA_DIR/bin:$PATH

# Conda環境の作成とパスの追加
#RUN conda create -y -n unique3d python=3.10
#ENV PATH=$CONDA_DIR/envs/unique3d/bin:$PATH

# Python の確認
RUN python3 --version

# ===============================
# Python ライブラリのインストール
# ===============================
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0  --index-url https://download.pytorch.org/whl/cu121

# 必要なパッケージを pip でインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# 
# kaolinインストール
#
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git && \
    cd kaolin && \
    git checkout v0.15.0 && \
    pip install .

# 作業ディレクトリ
WORKDIR /workspace

# ===============================
# エントリーポイント
# ===============================
CMD ["/bin/bash"]
