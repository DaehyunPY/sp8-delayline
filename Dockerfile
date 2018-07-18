FROM fedora
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

RUN dnf update -y \
    && dnf install -y python3 pipenv python3-root python3-jupyroot python3-jsmva \
    && dnf clean all

WORKDIR /app
COPY Pipefile Pipefile.lock /app/
RUN pipenv --python python3 --site-packages

ENTRYPOINT [ "pipenv", "run" ]
CMD python
