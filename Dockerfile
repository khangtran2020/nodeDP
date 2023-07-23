# Use nvidia/cuda image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion vim && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

COPY ./torch.yml /tmp/torch.yml
RUN conda update conda \
    && conda env create --name torch -f /tmp/torch.yml


RUN echo "conda activate torch" >> ~/.bashrc
ENV PATH /opt/conda/envs/pet/bin:$PATH
ENV CONDA_DEFAULT_ENV $torch

RUN adduser --disabled-password --gecos '' newuser \
    && adduser newuser sudo \
    && echo '%sudo ALL=(ALL:ALL) ALL' >> /etc/sudoers

WORKDIR /workspace/projects
RUN chown newuser ./

COPY ./ ./

RUN mkdir results
RUN mkdir results/dict
RUN mkdir results/models
RUN mkdir results/logs

USER newuser


