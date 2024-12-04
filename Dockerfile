FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    wget \
    git \
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
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile && \
    conda init bash

# build the conda environment
# ENV ENV_PREFIX $HOME/env

RUN conda update --name base -c defaults conda && \
    conda create --name raysam --no-default-packages python=3.11.9 && \
    conda clean --all --yes
    
# Activate the created dev env
SHELL ["conda", "run", "--no-capture-output", "-n", "raysam", "/bin/bash", "-c"]
RUN echo "source activate raysam" > ~/.bashrc

# ENV PATH /opt/conda/envs/dev/bin:$PATH
ENV PATH $HOME/miniconda3/envs/dev/bin:$PATH

RUN conda install -c anaconda pip \ 
    && conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes \
    && conda install -c conda-forge matplotlib \ 
        natsort python-xxhash segment-anything \
        python-elf kornia zarr pooch \
        pandas numpy pillow \
        #scikit-learn\
        scikit-image \
        pyyaml --yes \
    && conda clean --all --yes

    # h5py \
    # torchsummary \
    # einops \
    RUN pip install --no-cache-dir -U \
    timm \
    tensorboard \
    torch-tb-profiler \
    pyclean \
    imagecodecs \
    "ray[data,train,tune,serve]==2.24.0"


ENV SHELL=/bin/bash

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN env > /root/env.txt #&& cron -f

COPY torch_em/ $HOME/torch_em/
COPY micro_sam_ray/ $HOME/micro_sam_ray/
COPY run_sam_finetuning_hpa.py $HOME/run_sam_finetuning_hpa.py 
COPY download_datasets.py $HOME/download_datasets.py
ENTRYPOINT [ "/bin/bash" ]
# ENTRYPOINT [ "python", "run_sam_finetuning.py" ]
