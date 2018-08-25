FROM fedora
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

RUN dnf update -y \
    && dnf install -y \
        java which git pipenv \
        python3-root python3-jupyroot python3-jsmva \
    && dnf clean all

ENV LANG=en_US.UTF-8 \
    PATH=/root/.local/bin:$PATH \
    PYSPARK_PYTHON=/app/.venv/bin/python
COPY Pipfile Pipfile.lock install_spark_packages.py /app/
WORKDIR /app
# todo: fix --site-packages error
RUN pip3 install --user --upgrade pipenv \
    # && pipenv --three --site-packages \
    && pipenv --three \
    && pipenv install \
    && pipenv install --dev \
    && pipenv run /app/install_spark_packages.py \
    && rm -fr /root/.cache \
    && ln -s "$(pipenv --venv)" /app/.venv

EXPOSE 4040 8888
ENTRYPOINT [ "pipenv", "run" ]
CMD [ "python" ]
