FROM fedora
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

RUN dnf update -y \
    && dnf install -y which git python3 java \
                      python3-root python3-jupyroot python3-jsmva \
    && dnf clean all

WORKDIR /app
ENV LANG=en_US.UTF-8 \
    PIPENV_VENV_IN_PROJECT=true \
    PYSPARK_PYTHON=/app/.venv/bin/python
COPY Pipfile Pipfile.lock install_spark_packages.py /app/
RUN python3 -m pip install pipenv \
    && pipenv --python /usr/bin/python3 --site-packages \
    && pipenv install \
    && pipenv install --dev \
    && pipenv run /app/install_spark_packages.py \
    && rm -fr /root/.cache

ENTRYPOINT [ "pipenv", "run" ]
CMD [ "python" ]
