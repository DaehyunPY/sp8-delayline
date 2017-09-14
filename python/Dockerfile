FROM fedora:latest
MAINTAINER Daehyun You <daehyun.park.you@gmail.com>

USER root
WORKDIR /root

RUN dnf update -y && dnf install -y python3 python3-numpy python3-scipy python3-matplotlib python3-pandas \
                                    python3-cytoolz python3-PyYAML python3-h5py python3-tqdm \
                                    python3-root python3-jupyroot python3-jsmva \
    && dnf clean all
RUN pip3 install --no-cache-dir numba pathos

CMD [ "python3" ]
