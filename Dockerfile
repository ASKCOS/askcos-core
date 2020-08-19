ARG BASE_VERSION=2020.03.4-gh2855-py37-conda
ARG DATA_VERSION=dev

FROM registry.gitlab.com/mlpds_mit/askcos/askcos-data:$DATA_VERSION as data
FROM registry.gitlab.com/mlpds_mit/askcos/askcos-base:$BASE_VERSION

COPY --chown=askcos:askcos --from=data /data /usr/local/askcos-core/askcos/data
COPY --chown=askcos:askcos . /usr/local/askcos-core

WORKDIR /home/askcos

ENV PYTHONPATH=/usr/local/askcos-core:${PYTHONPATH}

LABEL core.version={VERSION} \
      core.git.hash={GIT_HASH} \
      core.git.date={GIT_DATE} \
      core.git.describe={GIT_DESCRIBE}
