FROM fedora
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

RUN dnf update -y \
    && dnf install -y \
        java which git pipenv \
        python3-root python3-jupyroot python3-jsmva \
    && dnf clean all

# todo: fix the code which is comment out
ENV LANG=en_US.UTF-8 \
    # PIPENV_VENV_IN_PROJECT=true \
    PATH=/root/.local/bin:$PATH \
    PYSPARK_PYTHON=/app/.venv/bin/python
# COPY Pipfile Pipfile.lock install_spark_packages.py /app/
COPY install_spark_packages.py /app/
WORKDIR /app
RUN pip3 install --user --upgrade pipenv \
        pyyaml numpy pandas scipy numba pyspark \
        matplotlib altair ipython jupyterlab \
        git+https://github.com/DaehyunPY/sp8-delayline.git \
    && pipenv --three --site-packages \
    # && pipenv install \
    # && pipenv install --dev \
    && pipenv run /app/install_spark_packages.py \
    && rm -fr /root/.cache \
    && ln -s "$(pipenv --venv)" /app/.venv

EXPOSE 4040 8888
ENTRYPOINT [ "pipenv", "run" ]
CMD [ "python" ]
