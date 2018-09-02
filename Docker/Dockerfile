FROM fedora
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

# install packages using dnf
ENV LANG=en_US.UTF-8
RUN dnf update -y \
    && dnf install -y \
        which bzip2 git java \
        python3-root python3-jupyroot python3-jsmva \
    && dnf clean all

# install Anaconda
WORKDIR /root
ENV PATH /opt/anaconda/bin:$PATH
ADD https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh anaconda.sh
RUN bash anaconda.sh -bp /opt/anaconda \
    && rm anaconda.sh

# build Pipenv project
WORKDIR /app
ENV PIPENV_VENV_IN_PROJECT=true \
    PATH=/root/.local/bin:/app/.venv/bin:$PATH \
    PYSPARK_PYTHON=/app/.venv/bin/python
COPY Pipfile install_spark_packages.py /app/
RUN pip install --user --upgrade pipenv \
    && pipenv --python /opt/anaconda/bin/python --site-packages \
    && pipenv install \
    && pipenv install --dev \
    && pipenv run /app/install_spark_packages.py \
    && rm -fr /root/.cache
COPY system.pth /app/.venv/lib/python3.6/site-packages/

EXPOSE 4040 8000 8888
ENTRYPOINT [ "pipenv", "run" ]
CMD [ "python" ]
