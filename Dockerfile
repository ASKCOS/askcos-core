ARG PY_VERSION=3.5-stretch
ARG RDKIT_VERSION=2019.03-py35
ARG DATA_VERSION=dev

FROM registry.gitlab.com/mlpds_mit/askcos/askcos/rdkit:$RDKIT_VERSION as rdkit

FROM registry.gitlab.com/mlpds_mit/askcos/makeit-data:$DATA_VERSION as data

FROM python:$PY_VERSION

COPY --from=rdkit /usr/local/rdkit-2019-03/rdkit /usr/local/rdkit-2019-03/rdkit
COPY --from=rdkit /usr/local/rdkit-2019-03/lib /usr/local/rdkit-2019-03/lib

RUN apt-get update && \
    apt-get install -y libboost-thread-dev libboost-python-dev libboost-iostreams-dev python-tk libopenblas-dev libeigen3-dev libcairo2-dev pkg-config python-dev python-mysqldb && \
    useradd -ms /bin/bash askcos

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

COPY --from=data /data /usr/local/askcos-core/makeit/data

COPY --chown=askcos:askcos . /usr/local/askcos-core

WORKDIR /home/askcos
USER askcos

ENV LD_LIBRARY_PATH=/usr/local/rdkit-2019-03/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/usr/local/askcos-core:/usr/local/rdkit-2019-03:${PYTHONPATH}

LABEL core.version={VERSION} \
      core.git.hash={GIT_HASH} \
      core.git.date={GIT_DATE} \
      core.git.describe={GIT_DESCRIBE}
