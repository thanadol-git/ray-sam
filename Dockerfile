FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ARG UID
# ARG GID
# ARG USER
# ARG GROUP

SHELL [ "/bin/bash", "--login", "-c" ]

# install utilities
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    apt-utils \
    curl \
    ca-certificates \
    sudo \
    wget \
    bzip2 \
    libx11-6 \
    ssh-client \
    bash-completion \
    libgl1-mesa-dev \
    tmux \
    screen \
    htop \
    cifs-utils \
    uidmap \
    nano \
    libjemalloc-dev \
    openssh-client \
    libpng-dev \
    libopenmpi-dev \
    mpich \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r raysam_user && useradd -r -g raysam_user raysam_user

# Add the user raysam_user to the sudo group
RUN echo "raysam_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV HOME /home/raysam_user

WORKDIR $HOME

# Change ownership of the application directory to the non-root user
RUN chown -R raysam_user:raysam_user $HOME

# switch to that user
USER $USER

# install miniconda
ENV MINICONDA_VERSION py312_24.9.2-0

ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O $HOME/miniconda.sh \
    && chmod +x $HOME/miniconda.sh \
    && $HOME/miniconda.sh -b -p $CONDA_DIR \
    && rm $HOME/miniconda.sh

# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH $CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# build the conda environment
# ENV ENV_PREFIX $HOME/env

RUN conda update --name base -c defaults conda \
    && conda create --name raysam python=3.11.9 \
    && conda clean --all --yes
    
# Activate the created dev env
SHELL ["conda", "run", "--no-capture-output", "-n", "raysam", "/bin/bash", "-c"]
RUN echo "source activate raysam" > ~/.bashrc

# ENV PATH /opt/conda/envs/dev/bin:$PATH
ENV PATH $HOME/miniconda3/envs/dev/bin:$PATH

RUN conda install -c anaconda pip \ 
    && conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes \
    && conda install -c conda-forge matplotlib \ 
        natsort python-xxhash segment-anything python-elf kornia zarr pooch \
        pandas numpy pillow \
        scikit-learn \
        scikit-image \
        pyyaml --yes \
    && conda clean --all --yes

RUN pip install --upgrade pip \
    && pip3 install --no-cache-dir jupyter jupyterlab \
    h5py \
    torchsummary \
    timm \
    tensorboard \
    einops \
    torch-tb-profiler \
    pyclean \
    opencv-python  
    # --upgrade scikit-learn

RUN pip install -U "ray[data,train,tune,serve]"

ENV SHELL=/bin/bash

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN env > /root/env.txt #&& cron -f

RUN /bin/bash -c "source activate raysam"

CMD ["/bin/bash"]
# CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]
