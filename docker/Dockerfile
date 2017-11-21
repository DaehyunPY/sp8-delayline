FROM fedora:latest

USER root
WORKDIR /root

RUN dnf update -y && \
    dnf install -y java python3-root python3-jupyroot python3-jsmva && \
    dnf clean all
COPY /requirements.txt /tmp
RUN pip3 install --no-cache-dir --requirement /tmp/requirements.txt

ENV PYSPARK_PYTHON /usr/bin/python3
ENV PYTHONPATH /root/lib
COPY /anatools /root/lib/anatools
COPY /export.py /root/lib

EXPOSE 4040
CMD [ "/usr/bin/python3" ]
