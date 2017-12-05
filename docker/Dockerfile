FROM fedora:latest

USER root
WORKDIR /root

RUN dnf update -y && \
    dnf install -y python3-root python3-jupyroot python3-jsmva && \
    dnf clean all
COPY /requirements.txt /tmp
RUN pip3 install --no-cache-dir --requirement /tmp/requirements.txt

ENV PYTHONPATH /root/lib
COPY /sp8tools /root/lib/sp8tools
COPY /export.py /root/lib

EXPOSE 4040
CMD [ "/usr/bin/python3" ]
