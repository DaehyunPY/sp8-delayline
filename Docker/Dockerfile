FROM fedora
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

RUN dnf update -y \
    && dnf install -y which python3 pipenv \
                      python3-root python3-jupyroot python3-jsmva \
                      java \
    && dnf clean all

WORKDIR /app
ENV LANG=en_US.UTF-8 \
    PYSPARK_PYTHON=/usr/bin/python3
COPY Pipfile Pipfile.lock /app/
RUN pipenv --python /usr/bin/python3 --site-packages \
    && pipenv install

ENTRYPOINT [ "pipenv", "run" ]
CMD [ "python" ]
