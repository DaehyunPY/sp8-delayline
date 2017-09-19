FROM fedora:latest

USER root
WORKDIR /root

RUN dnf update -y && \
    dnf install -y python3-root python3-jupyroot python3-jsmva && \
    dnf clean all
COPY /requirements.txt /tmp
RUN pip3 install --no-cache-dir --requirement /tmp/requirements.txt

ENV PYTHONPATH /root/lib
COPY /anatools /root/lib/anatools
COPY /export.py /root/lib

CMD [ "python3" ]
