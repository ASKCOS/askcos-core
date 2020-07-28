ARG BASE_VERSION=2019.03.4-gh2855-py35
ARG DATA_VERSION=dev

FROM registry.gitlab.com/mlpds_mit/askcos/askcos-data:$DATA_VERSION as data

FROM registry.gitlab.com/mlpds_mit/askcos/askcos-base:$BASE_VERSION

RUN apt-get update && \
    apt-get install -y libboost-thread-dev libboost-python-dev libboost-iostreams-dev python-tk libopenblas-dev libeigen3-dev libcairo2-dev pkg-config python-dev python-mysqldb && \
    useradd -ms /bin/bash askcos

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

COPY --from=data /data /usr/local/askcos-core/askcos/data

COPY --chown=askcos:askcos . /usr/local/askcos-core

WORKDIR /home/askcos
USER askcos

ENV PYTHONPATH=/usr/local/askcos-core:${PYTHONPATH}

LABEL core.version={VERSION} \
      core.git.hash={GIT_HASH} \
      core.git.date={GIT_DATE} \
      core.git.describe={GIT_DESCRIBE}
